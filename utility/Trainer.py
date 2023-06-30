import logging
import os
import gc
from itertools import chain

import pandas as pd
import torch
from torch import nn

from utility.Control import cfg
from utility.FunctionTime import timing_decorator
from utility.DataLoader import get_data_loaders
from utility.EverythingNeeded import get_memory_size_MB, print_gpu_info


class Trainer:
    def __init__(self, model, optimizer, lr_scheduler, loss_func, device, distributed=False):
        self.logger = logging.getLogger(self.__class__.__name__)

        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_func = loss_func
        self.device = device if torch.cuda.is_available() else cfg['device']
        self.summaries = None
        self.distributed = distributed
        self.rank = device
        self.acc_threshold = 0.5
        self.current_epoch = 0
        self.train_samples = 0
        self.valid_samples = 0
        # for combined loss
        self.prev_y_loss = 0
        self.prev_p_loss = 0
        self.y_weight = 0.5
        self.p_weight = 0.5

    def loss(self, y_loss_fn, y_pred, y_true, p_pred=None, p_true=None, weight=None):
        y_loss = y_loss_fn(y_pred, y_true, weight=weight)
        if cfg['momentum_predict']:
            loss_fn_p = nn.SmoothL1Loss()
            p_loss = loss_fn_p(p_pred, p_true)

            # calculate rates of decrease
            y_rate = abs(self.prev_y_loss - y_loss) if self.prev_y_loss else torch.tensor(0.)
            p_rate = abs(self.prev_p_loss - p_loss) if self.prev_p_loss else torch.tensor(0.)

            # calculate new weights based on rates of decrease
            # (add small constant in denominator to avoid division by zero)
            self.y_weight = y_rate / (y_rate + p_rate + 1e-10)
            self.p_weight = 1 - self.y_weight

            # update previous losses
            self.prev_y_loss = y_loss
            self.prev_p_loss = p_loss

            return self.y_weight.item() * y_loss + self.p_weight.item() * p_loss
        else:
            return y_loss

    @timing_decorator
    def process(self, n_epochs, n_total_epochs, world_size):
        # Determine initial epoch in case resuming training
        start_epoch = 0
        if self.summaries is not None:
            start_epoch = self.summaries.epoch.max() + 1

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

                train_sum = self.train_iteration(train_data)
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
    def train_iteration(self, data_loader):
        """Train for one epoch"""
        self.model.train()

        # Prepare summary information
        summary = dict()
        sum_loss = 0

        # Loop over training batches
        for i, batch in enumerate(data_loader):
            if torch.cuda.is_available():
                # Print memory usage at the start of each batch
                self.logger.debug(f'[Batch {i}] Memory allocated: {torch.cuda.memory_allocated() / (1024 * 1024)} MB')
                self.logger.debug(f'[Batch {i}] Memory reserved: {torch.cuda.memory_reserved() / (1024 * 1024)} MB')
                print_gpu_info(self.logger)
            self.logger.debug(f'[Batch {i}] Batch size: {get_memory_size_MB(batch)} MB')

            self.train_samples += batch.num_graphs
            batch = batch.to(self.device)
            if torch.cuda.is_available():
                print_gpu_info(self.logger)
            self.model.zero_grad()
            batch_out = self.model(batch, verbose=True if get_memory_size_MB(batch) > 1 else False)

            if torch.cuda.is_available():
                print_gpu_info(self.logger)
                # Print memory usage at the start of each batch
                self.logger.debug(
                    f'[Batch {i}] Model Output Memory allocated: {torch.cuda.memory_allocated() / (1024 * 1024)} MB')
                self.logger.debug(
                    f'[Batch {i}] Model Output Memory reserved: {torch.cuda.memory_reserved() / (1024 * 1024)} MB')

            if cfg['momentum_predict']:
                y_pred, p_out = batch_out
                # calculate momentum prediction
                con_mask = (batch.y == 1)
                p_truth = batch.p[con_mask]
                p_pred = p_out[con_mask]
                del con_mask
            else:
                y_pred = batch_out
                p_truth, p_pred = None, None

            batch_loss = self.loss(self.loss_func, y_pred, batch.y, p_pred, p_truth, weight=batch.w)

            batch_loss.backward()

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
        sum_correct = 0
        sum_total = 0
        sum_tp, sum_fp, sum_fn, sum_tn = 0, 0, 0, 0
        diff_list = []

        if large_loader is not None:
            self.logger.debug('Running validation on large loader with size %i', len(large_loader))
        # Loop over batches
        for i, batch in enumerate(data_loader if large_loader is None else chain(data_loader, large_loader)):
            self.logger.debug(f'[Batch {i}] Batch size: {get_memory_size_MB(batch)} MB')

            self.valid_samples += batch.num_graphs

            batch = batch.to(self.device)

            batch_out = self.model(batch)
            if cfg['momentum_predict']:
                y_pred, p_out = batch_out
                con_mask = (batch.y == 1)
                p_truth = batch.p[con_mask]
                p_pred = p_out[con_mask]
            else:
                y_pred = batch_out
                p_truth, p_pred = None, None

            batch_loss = self.loss(self.loss_func, y_pred, batch.y, p_pred, p_truth, weight=batch.w).item()
            sum_loss += batch_loss

            # Count number of correct predictions
            batch_pred = torch.sigmoid(y_pred)
            batch_pred = batch_pred > self.acc_threshold
            truth_label = batch.y > self.acc_threshold
            matches = (batch_pred == truth_label)
            sum_correct += matches.sum().item()
            sum_total += matches.numel()
            # Compute true positives, false positives, true negatives, and false negatives
            sum_tp += ((batch_pred == 1) & (truth_label == 1)).sum().item()
            sum_fp += ((batch_pred == 1) & (truth_label == 0)).sum().item()
            sum_tn += ((batch_pred == 0) & (truth_label == 0)).sum().item()
            sum_fn += ((batch_pred == 0) & (truth_label == 1)).sum().item()
            # Count the difference between truth p and predicted p
            if cfg['momentum_predict']:
                p_err = (p_pred - p_truth) / p_truth

                # Count the number of NaN values
                nan_count = torch.isnan(p_err).sum()
                self.logger.debug(f"Number of NaN values: {nan_count.item()}")

                # Count the number of Inf values
                inf_count = torch.isinf(p_err).sum()
                self.logger.debug(f"Number of Inf values: {inf_count.item()}")

                finite_mask = torch.isfinite(p_err)

                diff_list.append(p_err[finite_mask])
            self.logger.debug(' valid batch %i, loss %.4f', i, batch_loss)

            # del batch, batch_out, batch_loss
            # del y_pred, p_out, p_pred, p_truth
            # if torch.cuda.is_available():
            #     torch.cuda.empty_cache()
            # gc.collect()

        # Summarize the validation epoch
        n_batches = len(data_loader)
        diff = torch.cat(diff_list, dim=0) if cfg['momentum_predict'] else torch.Tensor([-999])
        summary['valid_loss'] = sum_loss / n_batches
        summary['valid_acc'] = sum_correct / sum_total
        summary["valid_TP"] = sum_tp
        summary["valid_FP"] = sum_fp
        summary["valid_TN"] = sum_tn
        summary["valid_FN"] = sum_fn
        summary['valid_batches'] = n_batches
        summary['valid_sum_total'] = sum_total
        summary['valid_dp_mean'] = diff.mean(dim=0).item()
        summary['valid_dp_std'] = diff.std(dim=0).item()

        # # Check for NaN values
        # has_nan = torch.isnan(diff).any()
        # print(f"Contains NaN values: {has_nan.item()}")
        #
        # # Check for Inf values
        # has_inf = torch.isinf(diff).any()
        # print(f"Contains Inf values: {has_inf.item()}")

        self.logger.debug(' Processed %i samples in %i batches', len(data_loader.sampler), n_batches)
        self.logger.debug(' -- momentum mean %.3f std %.3f ' % (summary['valid_dp_mean'], summary['valid_dp_std']))
        self.logger.info('  Validation loss: %.3f acc: %.3f' % (summary['valid_loss'], summary['valid_acc']))

        return summary

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
        norm += p.grad.data.norm(norm_type).item() ** norm_type
    return norm ** (1. / norm_type)
