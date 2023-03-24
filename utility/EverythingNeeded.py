import logging
import math
import os
import sys
from functools import partial

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

import models
from utility.Control import cfg


def build_model(rank, distributed=False):
    if 'model' in cfg:
        model_configs = cfg['model']
        name = model_configs.pop('name')
        model = models.get_model(name=name, **model_configs).to(rank)

        # print(model)
        print('Parameters: %i' % sum(p.numel() for p in model.parameters()))

        if distributed:
            return DistributedDataParallel(model, device_ids=[rank])
        else:
            return model
    else:
        print("model is not missing in config.")
        return None


def build_optimizer(parameters, name='Adam', learning_rate=0.001,
                    lr_warmup_epochs=0, lr_decay_schedule=None,
                    **optimizer_args):
    """Construct the training optimizer and scale learning rate.
    Should be called by build_model rather than called directly."""

    # Compute the scaled learning rate and corresponding initial warmup factor
    if lr_decay_schedule is None:
        lr_decay_schedule = []

    # Construct the optimizer
    optimizer_type = getattr(torch.optim, name)
    optimizer = optimizer_type(parameters, lr=learning_rate, **optimizer_args)

    # Prepare the learning rate scheduler
    def _lr_schedule(epoch, warmup_factor=1, warmup_epochs=0, decays=None):
        if decays is None:
            decays = []
        if epoch < warmup_epochs:
            return (1 - warmup_factor) * epoch / warmup_epochs + warmup_factor
        for decay in decays:
            if decay['start_epoch'] <= epoch < decay['end_epoch']:
                return decay['factor']
        return 1

    lr_schedule = partial(
        _lr_schedule, warmup_factor=1,
        warmup_epochs=lr_warmup_epochs, decays=lr_decay_schedule
    )
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)
    return optimizer, lr_scheduler


def build_loss(name):
    loss_func = getattr(nn.functional, name)
    return loss_func


def config_logging(verbose, output_dir, append=False, rank=0):
    log_format = '%(asctime)s %(levelname)s %(message)s'
    log_level = logging.DEBUG if verbose else logging.INFO
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(log_level)
    handlers = [stream_handler]
    if output_dir is not None:
        log_dir = output_dir
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, 'out_%i.log' % rank)
        mode = 'a' if append else 'w'
        file_handler = logging.FileHandler(log_file, mode=mode)
        file_handler.setLevel(log_level)
        handlers.append(file_handler)
    logging.basicConfig(level=log_level, format=log_format, handlers=handlers)
    # Suppress annoying matplotlib debug printouts
    logging.getLogger('matplotlib').setLevel(logging.ERROR)
