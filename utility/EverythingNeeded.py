import logging
import math
import os
import sys
from functools import partial

import pandas as pd
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

import models
from utility.Control import cfg


def build_model(rank, distributed=False):
    if 'model' in cfg:
        model_configs = cfg['model']
        model = models.get_model(**model_configs).to(rank)

        # print(model)
        print('Parameters: %i' % sum(p.numel() for p in model.parameters()))

        if distributed:
            return DistributedDataParallel(model, device_ids=[rank])
        else:
            return model
    else:
        print("model is missing in config.")
        return None


def build_optimizer(parameters, n_rank, name='Adam', learning_rate=0.001,
                    lr_warmup_epochs=1, lr_decay_schedule=None, lr_scaling='sqrt',
                    **optimizer_args):
    """Construct the training optimizer and scale learning rate.
    Should be called by build_model rather than called directly."""

    # Compute the scaled learning rate and corresponding initial warmup factor
    if lr_decay_schedule is None:
        lr_decay_schedule = []

    # Compute the scaled learning rate and corresponding initial warmup factor
    warmup_factor = 1
    if lr_scaling == 'linear':
        learning_rate = learning_rate * n_rank
        warmup_factor = 1. / n_rank
    elif lr_scaling == 'sqrt':
        learning_rate = learning_rate * math.sqrt(n_rank)
        warmup_factor = 1. / math.sqrt(n_rank)

    # Construct the optimizer
    optimizer_type = getattr(torch.optim, name)
    optimizer = optimizer_type(parameters, lr=learning_rate, **optimizer_args)

    # Prepare the learning rate scheduler
    def _lr_schedule(epoch, warm_up_factor=1, warmup_epochs=0, decays=None):
        if decays is None:
            decays = []
        if epoch < warmup_epochs:
            return (1 - warm_up_factor) * epoch / warmup_epochs + warm_up_factor
        for decay in decays:
            if decay['start_epoch'] <= epoch < decay['end_epoch']:
                return decay['factor']
        return 1

    lr_schedule = partial(
        _lr_schedule, warm_up_factor=warmup_factor,
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


def get_item_from_dataloader(dataloader, index):
    data_iterator = iter(dataloader)
    for idx, batch in enumerate(data_iterator):
        if idx == index:
            return batch
    raise StopIteration("Index out of range")


def convert_batch_to_df(batch):
    # Convert node features (x) to a DataFrame
    df_node = pd.DataFrame(batch.x.cpu().numpy(), columns=['x', 'y', 'z'])

    # Convert edge_index to a DataFrame with 'start' and 'end' columns
    edge_index_numpy = batch.edge_index.cpu().numpy()
    df_edge = pd.DataFrame(edge_index_numpy.T, columns=['start', 'end'])

    # Convert target (y) to a DataFrame
    df_y = pd.DataFrame(batch.y.cpu().numpy(), columns=['truth'])

    return {
        'node': df_node,
        'edge': df_edge,
        'y': df_y,
    }
