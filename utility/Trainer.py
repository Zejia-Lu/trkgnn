import logging
import time

import torch

from utility.Control import cfg
from utility.FunctionTime import timing_decorator
from utility.DataLoader import get_data_loaders


class Trainer:
    def __init__(self, model, optimizer, lr_scheduler, loss_func, device, distributed=False):
        self.logger = logging.getLogger(self.__class__.__name__)

        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_func = loss_func
        self.device = device
        self.summaries = None
        self.distributed = distributed

    def process(self, n_epochs, n_total_epochs, rank, world_size):
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

            # Train on this epoch
            self.process_epoch(epoch, rank, world_size)
            self.lr_scheduler.step()

            # Save summary, checkpoint
            # self.save_summary(summary)
            # if self.output_dir is not None and self.rank == 0:
            #     self.write_checkpoint(checkpoint_id=epoch)

    @timing_decorator
    def process_epoch(self, epoch, rank, world_size):
        data_generator = get_data_loaders(
            cfg['data']['input_dir'],
            chunk_size=10,
            batch_size=2,
            distributed=self.distributed,
            n_workers=cfg['data']['n_workers'],
            rank=rank,
            n_ranks=world_size
        )
        while True:
            try:
                train_data, valid_data = next(data_generator)
                try:
                    train_data.sampler.set_epoch(epoch)
                except AttributeError:
                    pass

                train_sum = self.train_iteration(train_data)
                valid_sum = self.valid_iteration(valid_data)

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
            batch = batch.to(self.device)
            self.model.zero_grad()
            batch_output = self.model(batch)
            batch_loss = self.loss_func(batch_output, batch.y, weight=batch.w)

            batch_loss.backward()
            self.optimizer.step()
            sum_loss += batch_loss.item()

            # Dump additional debugging information
            if self.logger.isEnabledFor(logging.DEBUG):
                l1 = get_weight_norm(self.model, 1)
                l2 = get_weight_norm(self.model, 2)
                grad_norm = get_grad_norm(self.model)
                self.logger.debug('  train batch %i loss %.4f l1 %.2f l2 %.4f grad %.3f idx %i',
                                  i, batch_loss.item(), l1, l2, grad_norm, batch.i[0].item())

        # Summarize the epoch
        n_batches = i + 1
        summary['lr'] = self.optimizer.param_groups[0]['lr']
        summary['train_loss'] = sum_loss / n_batches
        summary['l1'] = get_weight_norm(self.model, 1)
        summary['l2'] = get_weight_norm(self.model, 2)
        self.logger.debug(' Processed %i batches', n_batches)
        self.logger.debug(' Model LR %f l1 %.2f l2 %.2f', summary['lr'], summary['l1'], summary['l2'])
        self.logger.info('  Training loss: %.3f', summary['train_loss'])
        return summary

    @timing_decorator
    @torch.no_grad()
    def valid_iteration(self, data_loader):
        """"Evaluate the model"""
        self.model.eval()

        # Prepare summary information
        summary = dict()
        sum_loss = 0
        sum_correct = 0
        sum_total = 0

        # Loop over batches
        for i, batch in enumerate(data_loader):
            batch = batch.to(self.device)

            # Make predictions on this batch
            batch_output = self.model(batch)
            batch_loss = self.loss_func(batch_output, batch.y).item()
            sum_loss += batch_loss

            # Count number of correct predictions
            batch_pred = torch.sigmoid(batch_output)
            matches = ((batch_pred > 0.5) == (batch.y > 0.5))
            sum_correct += matches.sum().item()
            sum_total += matches.numel()
            self.logger.debug(' valid batch %i, loss %.4f', i, batch_loss)

        # Summarize the validation epoch
        n_batches = i + 1
        summary['valid_loss'] = sum_loss / n_batches
        summary['valid_acc'] = sum_correct / sum_total
        self.logger.debug(' Processed %i samples in %i batches', len(data_loader.sampler), n_batches)
        self.logger.info('  Validation loss: %.3f acc: %.3f' % (summary['valid_loss'], summary['valid_acc']))
        return summary


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
