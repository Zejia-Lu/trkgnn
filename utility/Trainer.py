import logging
import os
import gc
from itertools import chain

import pandas as pd
import torch
from torch import nn
import torch.distributed as dist
from torch.autograd import grad
import torch.nn.functional as F

from utility.Control import cfg
from utility.FunctionTime import timing_decorator
from utility.DataLoader import get_data_loaders
from utility.EverythingNeeded import get_memory_size_MB, print_gpu_info, cluster_graphs


class Trainer:
    def __init__(
            self, model, optimizer,
            lr_scheduler: torch.optim.lr_scheduler.LambdaLR,
            loss_func,
            device,
            distributed=False):
        self.logger = logging.getLogger(self.__class__.__name__)

        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_func_y = loss_func
        self.loss_func_p = nn.SmoothL1Loss(beta=0.1)
        self.device = device if torch.cuda.is_available() else cfg['device']
        self.summaries = None
        self.distributed = distributed
        self.rank = device
        self.acc_threshold = 0.5
        self.current_epoch = 0
        self.train_samples = 0
        self.valid_samples = 0
        # Alpha for Grad-Norm
        self.alpha = 0.25
        self.min_factor = 1e-3
        # Initialize gradient norms for each task
        self.G_0 = None
        # Initial weights for each task
        self.weights = torch.tensor([1.0, 1.0], device=self.device)

    def y_loss(self, y_pred, y_true, weight=None):
        loss = self.loss_func_y(y_pred, y_true, weight=weight)
        return loss

    def p_loss(self, p_pred, p_true):
        return self.loss_func_p(p_pred, p_true)

    @timing_decorator
    def process(self, n_epochs, n_total_epochs, world_size):
        # Determine initial epoch in case resuming training
        start_epoch = 0
        if self.summaries is not None:
            start_epoch = self.summaries['epoch'].iloc[-1] + 1

        # Determine how many epochs we run in this call
        end_epoch = n_total_epochs
        if n_epochs >= 0:
            end_epoch = min(start_epoch + n_epochs, n_total_epochs)

        self.logger.debug('Will train epochs %i - %i', start_epoch, end_epoch)

        # Loop over epochs
        for epoch in range(start_epoch, end_epoch):
            self.logger.info('Epoch %i' % epoch)
            self.current_epoch = epoch

            self.train_samples = 0
            self.valid_samples = 0

            # Train on this epoch
            self.process_epoch(epoch, world_size)
            if self.rank == 0:
                self.write_checkpoint(epoch)
            self.lr_scheduler.step()

            self.logger.info(
                f'Epoch {epoch} trained {self.train_samples} samples and validated {self.valid_samples} samples')

        # Save summary, checkpoint
        self.save_summary()
        # if self.output_dir is not None and self.rank == 0:
        #     self.write_checkpoint(checkpoint_id=epoch)

    @timing_decorator
    def process_epoch(self, epoch, world_size):
        data_input_dir = cfg['data']['input_graph_dir'] if cfg['data']['read_from_graph'] else cfg['data']['input_dir']
        data_generator = get_data_loaders(
            data_input_dir,
            chunk_size=cfg['data']['chunk_size'],
            batch_size=cfg['data']['batch_size'],
            distributed=self.distributed,
            n_workers=cfg['data']['n_workers'],
            rank=self.rank,
            n_ranks=world_size,
        )
        itr = 0
        while True:
            if torch.cuda.is_available():
                # Print memory usage at the start of each batch
                self.logger.debug(
                    f'[Iteration: {itr}] Memory allocated: {torch.cuda.memory_allocated() / (1024 * 1024)} MB')
                self.logger.debug(
                    f'[Iteration: {itr}] Memory reserved: {torch.cuda.memory_reserved() / (1024 * 1024)} MB')

            try:
                train_data, valid_data, large_data = next(data_generator)
                try:
                    train_data.sampler.set_epoch(epoch)
                except AttributeError:
                    self.logger.info('Sampler has no set_epoch method')
                    pass

                train_sum = self.train_iteration(train_data, large_data)
                valid_sum = self.valid_iteration(valid_data, large_data)

                train_sum['itr'] = itr
                train_sum['epoch'] = epoch

                df_sum = pd.concat([pd.DataFrame(s, index=[0]) for s in [train_sum, valid_sum]], axis=1)
                self.add_summary(df_sum)
                self.save_summary()

                del train_sum, valid_sum, df_sum
                del train_data, valid_data
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

                if torch.cuda.is_available():
                    # Print memory usage at the start of each batch
                    self.logger.info(
                        f'[Iteration: {itr}] Peak Memory allocated: {torch.cuda.max_memory_allocated() / (1024 * 1024)} MB')
                    self.logger.info(
                        f'[Iteration: {itr}] Peak Memory reserved: {torch.cuda.max_memory_reserved() / (1024 * 1024)} MB')

                itr += 1
            except StopIteration:
                print("Finish")
                break

    @timing_decorator
    def train_iteration(self, data_loader, large_loader=None):
        """Train for one epoch"""

        self.model.train()

        # Prepare summary information
        summary = dict()
        sum_loss = 0
        batch_loss = None

        # Loop over training batches
        for i, batch in enumerate(data_loader if large_loader is None else chain(data_loader, large_loader)):

            self.logger.debug(f'[Batch {i}] Batch size: {get_memory_size_MB(batch)} MB')

            self.train_samples += batch.num_graphs
            batch = batch.to(self.device)
            if torch.cuda.is_available():
                print_gpu_info(self.logger)
            self.model.zero_grad()
            batch_out = self.model(batch)

            if cfg['task'] == 'link':
                y_pred = batch_out
                batch_loss = self.loss_func_y(y_pred, batch.y, weight=batch.w)
                batch_loss.backward()
            elif cfg['task'] == 'momentum':
                p_pred_all, new_y_score = batch_out
                # calculate momentum prediction
                con_mask = (batch.y == 1)
                p_truth = batch.p[con_mask]
                p_pred = p_pred_all[con_mask]
                batch_loss = self.loss_func_p(p_pred, p_truth)
                del con_mask
                batch_loss.backward()

            if batch_loss is not None:
                self.optimizer.step()
                sum_loss += batch_loss.item()

            # Dump additional debugging information
            if self.logger.isEnabledFor(logging.DEBUG):
                l1 = get_weight_norm(self.model, 1)
                l2 = get_weight_norm(self.model, 2)
                grad_norm = get_grad_norm(self.model)
                self.logger.debug(
                    '  train batch %i loss %.4f l1 %.2f l2 %.4f grad %.3f idx %i',
                    i, batch_loss.item(), l1, l2, grad_norm, batch.i[0].item()
                )
                del l1, l2, grad_norm

            if torch.cuda.is_available():
                # Print memory usage at the start of each batch
                self.logger.debug(
                    f'[After Batch {i}] Memory allocated: {torch.cuda.memory_allocated() / (1024 * 1024)} MB')
                self.logger.debug(
                    f'[After Batch {i}] Memory reserved: {torch.cuda.memory_reserved() / (1024 * 1024)} MB')
                # After processing a batch, print the peak memory usage
                self.logger.debug(
                    f'[After Batch {i}] Peak memory allocated: {torch.cuda.max_memory_allocated() / (1024 * 1024)} MB')
                self.logger.debug(
                    f'[After Batch {i}] Peak memory reserved: {torch.cuda.max_memory_reserved() / (1024 * 1024)} MB')

            # del batch, batch_out, batch_loss
            # del y_pred, p_out, p_pred, p_truth
            # if torch.cuda.is_available():
            #     torch.cuda.empty_cache()
            # gc.collect()

        # Summarize the epoch
        n_batches = len(data_loader)
        summary['lr'] = self.optimizer.param_groups[0]['lr']
        summary['train_loss'] = sum_loss / n_batches
        summary['l1'] = get_weight_norm(self.model, 1)
        summary['l2'] = get_weight_norm(self.model, 2)
        summary['grad_norm'] = get_grad_norm(self.model)
        summary['train_batches'] = n_batches
        self.logger.debug(' Processed %i batches', n_batches)
        self.logger.debug(' Model LR %f l1 %.2f l2 %.2f', summary['lr'], summary['l1'], summary['l2'])
        self.logger.info('  Training loss: %.3f', summary['train_loss'])
        return summary

    @timing_decorator
    @torch.no_grad()
    def valid_iteration(self, data_loader, large_loader=None):
        """"Evaluate the model"""
        self.model.eval()

        # Prepare summary information
        summary = dict()
        sum_loss = 0

        results_link = {
            'sum_correct': 0,
            'sum_total': 0,
            'sum_tp': 0,
            'sum_fp': 0,
            'sum_fn': 0,
            'sum_tn': 0,
        }

        results_momentum = {

        }

        diff_list = []
        num_tracks_diff_list = {
            0: 0,
            1: 0,
            2: 0,
            3: 0,
            -1: 0,
            -2: 0,
            -3: 0,
        }

        batch_loss = None

        if large_loader is not None:
            self.logger.debug('Running validation on large loader with size %i', len(large_loader))
        # Loop over batches
        for i, batch in enumerate(data_loader if large_loader is None else chain(data_loader, large_loader)):
            self.logger.debug(f'[Batch {i}] Batch size: {get_memory_size_MB(batch)} MB, with length {len(batch)}')

            self.valid_samples += batch.num_graphs

            batch = batch.to(self.device)

            batch_out = self.model(batch)

            if cfg['task'] == 'link':
                y_pred = batch_out
                batch_loss = self.loss_func_y(y_pred, batch.y, weight=batch.w)

                self.eval_link(y_pred, batch.y, results_link, weight=batch.w)

                if cfg['num_track_predict']:
                    # # DBSCAN for graphs
                    num_tracks = cluster_graphs(batch, y_pred, eps=self.acc_threshold)

                    n_pred = num_tracks
                    n_truth = batch.n.detach().cpu().numpy()

                    # Count the difference between truth n and predicted n
                    n_diff = (n_pred - n_truth)
                    # classify difference into num_tracks_diff_list
                    num_tracks_diff_list[0] += len(n_diff[n_diff == 0])
                    num_tracks_diff_list[1] += len(n_diff[n_diff == 1])
                    num_tracks_diff_list[2] += len(n_diff[n_diff == 2])
                    num_tracks_diff_list[-1] += len(n_diff[n_diff == -1])
                    num_tracks_diff_list[-2] += len(n_diff[n_diff == -2])
                    num_tracks_diff_list[-3] += len(n_diff[n_diff < -2])
                    num_tracks_diff_list[3] += len(n_diff[n_diff > 2])

            elif cfg['task'] == 'momentum':
                p_pred_all, new_y_score = batch_out
                # calculate momentum prediction
                con_mask = (batch.y == 1)
                p_truth = batch.p[con_mask]
                p_pred = p_pred_all[con_mask]
                batch_loss = self.loss_func_p(p_pred, p_truth)
                del con_mask

                self.eval_momentum(p_pred, p_truth, diff_list)

            if batch_loss is not None:
                sum_loss += batch_loss.item()

            self.logger.debug(' valid batch %i, loss %.4f', i, batch_loss)

            # del batch, batch_out, batch_loss
            # del y_pred, p_out, p_pred, p_truth
            # if torch.cuda.is_available():
            #     torch.cuda.empty_cache()
            # gc.collect()

        # Summarize the validation epoch
        n_batches = len(data_loader)
        summary['valid_loss'] = sum_loss / n_batches
        summary['valid_batches'] = n_batches

        if cfg['task'] == 'link':
            summary['valid_acc'] = results_link['sum_correct'] / results_link['sum_total']
            summary["valid_TP"] = results_link['sum_tp']
            summary["valid_FP"] = results_link['sum_fp']
            summary["valid_TN"] = results_link['sum_tn']
            summary["valid_FN"] = results_link['sum_fn']
            summary['valid_sum_total'] = results_link['sum_total']

            if cfg['num_track_predict']:
                summary['valid_num_tracks_diff_0'] = num_tracks_diff_list[0]
                summary['valid_num_tracks_diff_1'] = num_tracks_diff_list[1]
                summary['valid_num_tracks_diff_2'] = num_tracks_diff_list[2]
                summary['valid_num_tracks_diff_-1'] = num_tracks_diff_list[-1]
                summary['valid_num_tracks_diff_-2'] = num_tracks_diff_list[-2]
                summary['valid_num_tracks_diff_underflow'] = num_tracks_diff_list[-3]
                summary['valid_num_tracks_diff_overflow'] = num_tracks_diff_list[3]

        if cfg['task'] == 'momentum':
            diff = torch.cat(diff_list, dim=0)
            summary['valid_dp_mean'] = diff.mean(dim=0).item()
            summary['valid_dp_std'] = diff.std(dim=0).item()

            self.logger.debug(' -- momentum mean %.3f std %.3f ' % (summary['valid_dp_mean'], summary['valid_dp_std']))

        # # Check for NaN values
        # has_nan = torch.isnan(diff).any()
        # print(f"Contains NaN values: {has_nan.item()}")
        #
        # # Check for Inf values
        # has_inf = torch.isinf(diff).any()
        # print(f"Contains Inf values: {has_inf.item()}")

        self.logger.debug(' Processed %i samples in %i batches', len(data_loader.sampler), n_batches)
        self.logger.info('  Validation loss: %.3f' % summary['valid_loss'])

        return summary

    def eval_link(self, y_pred, y_true, results, weight=None):
        # Count number of correct predictions
        batch_pred = torch.sigmoid(y_pred)
        batch_pred = batch_pred > self.acc_threshold
        truth_label = y_true > self.acc_threshold

        matches = (batch_pred == truth_label)
        results['sum_correct'] += matches.float().mul(weight).sum().item()
        results['sum_total'] += weight.sum().item()
        self.logger.debug('correct: %i, total: %i', results['sum_correct'], results['sum_total'])

        # Compute weighted true positives, false positives, true negatives, and false negatives
        results['sum_tp'] += ((batch_pred == 1) & (truth_label == 1)).float().mul(weight).sum().item()
        results['sum_fp'] += ((batch_pred == 1) & (truth_label == 0)).float().mul(weight).sum().item()
        results['sum_tn'] += ((batch_pred == 0) & (truth_label == 0)).float().mul(weight).sum().item()
        results['sum_fn'] += ((batch_pred == 0) & (truth_label == 1)).float().mul(weight).sum().item()

    def eval_momentum(self, p_pred, p_truth, diff_list):
        p_err = (p_pred - p_truth) / p_truth

        # Count the number of NaN values
        nan_count = torch.isnan(p_err).sum()
        self.logger.debug(f"Number of NaN values: {nan_count.item()}")

        # Count the number of Inf values
        inf_count = torch.isinf(p_err).sum()
        self.logger.debug(f"Number of Inf values: {inf_count.item()}")

        finite_mask = torch.isfinite(p_err)

        diff_list.append(p_err[finite_mask])

    def add_summary(self, summaries):
        if self.summaries is None:
            self.summaries = summaries
        else:
            self.summaries = pd.concat([self.summaries, summaries], ignore_index=True)

    def save_summary(self):
        if cfg['output_dir']:
            summary_file = os.path.join(cfg['output_dir'], 'summaries_%i.csv' % self.rank)
            self.summaries.to_csv(summary_file, index=False)
            self.logger.info(f'[Rank {self.rank}]: Write summary to {summary_file}')
        pass

    def write_checkpoint(self, checkpoint_id):
        """Write a checkpoint for the model"""
        assert cfg['output_dir'] is not None
        # If using DistributedDataParallel, just save the wrapped model state
        if self.distributed:
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()
        checkpoint = dict(
            checkpoint_id=checkpoint_id,
            model=model_state_dict,
            optimizer=self.optimizer.state_dict(),
            lr_scheduler=self.lr_scheduler.state_dict()
        )
        checkpoint_dir = os.path.join(cfg['output_dir'], 'model.checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_file = 'model_checkpoint_%03i.pth.tar' % checkpoint_id
        torch.save(checkpoint, os.path.join(checkpoint_dir, checkpoint_file))
        self.logger.info(f'[Rank {self.rank}]: Write checkpoint {checkpoint_id} to {checkpoint_file}')


def get_weight_norm(model, norm_type=2):
    """Get the norm of the model weights"""
    norm = 0
    for p in model.parameters():
        norm += p.data.norm(norm_type).item() ** norm_type
    return norm ** (1. / norm_type)


def get_grad_norm(model, norm_type=2):
    """Get the norm of the model weight gradients"""
    norm = 0
    for p in model.parameters():
        if p.grad is not None:
            norm += p.grad.data.norm(norm_type).item() ** norm_type
    return norm ** (1. / norm_type)
