import logging
import math
import os
import subprocess
import sys
from functools import partial

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch_geometric.data import Data

from sklearn.cluster import DBSCAN

import models
from utility.Control import cfg
from utility.FunctionTime import timing_decorator


def build_model(rank, distributed=False, existed_model_path: str = None):
    logger = logging.getLogger(__name__)
    if 'model' in cfg:
        model_configs = cfg['model']
        model = models.get_model(**model_configs).to(rank)

        if existed_model_path is not None:
            model.load_state_dict(torch.load(existed_model_path)['model'])
            logger.info(f"Loaded model from {existed_model_path}")

        # print(model)
        logger.debug('Parameters: %i' % sum(p.numel() for p in model.parameters()))

        if distributed:
            return DistributedDataParallel(model, device_ids=[rank], static_graph=False)
        else:
            # return torch.compile(model, mode="reduce-overhead")
            return model
    else:
        logger.error("model is missing in config.")
        return None


def build_optimizer(parameters, n_rank, name='Adam', learning_rate=0.001,
                    lr_warmup_epochs=1, lr_decay_schedule=None, lr_scaling='sqrt',
                    existed_optimizer_path=None,
                    last_epoch=-1,
                    **optimizer_args):
    """Construct the training optimizer and scale learning rate.
    Should be called by build_model rather than called directly."""
    logger = logging.getLogger(__name__)

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

    if existed_optimizer_path is not None:
        optimizer.load_state_dict(torch.load(existed_optimizer_path)['optimizer'])
        logger.info(f"Loaded optimizer from {existed_optimizer_path}")
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule, last_epoch=last_epoch)
    return optimizer, lr_scheduler


def build_loss(name):
    loss_func = getattr(nn.functional, name)
    return loss_func


def config_logging(verbose, output_dir, append=False, rank=0, prefix='out'):
    log_format = '%(asctime)s - [%(name)s: %(levelname)s] %(message)s'
    log_level = logging.DEBUG if verbose else logging.INFO
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(log_level)
    handlers = [stream_handler]
    if output_dir is not None:
        log_dir = output_dir
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f'{prefix}_%i.log' % rank)
        mode = 'a' if append else 'w'
        file_handler = logging.FileHandler(log_file, mode=mode)
        file_handler.setLevel(log_level)
        handlers.append(file_handler)
    logging.basicConfig(level=log_level, format=log_format, handlers=handlers)
    # Suppress annoying matplotlib debug printouts
    # logging.getLogger('matplotlib').setLevel(logging.ERROR)


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


def get_memory_size_MB(data: torch.tensor):
    total_memory = 0
    for attr, value in data:
        if torch.is_tensor(value):
            # itemsize gives memory size per element
            # numel gives number of elements
            total_memory += value.numel() * value.element_size()

    return total_memory / (1024 * 1024)


def print_gpu_info(logger=None, prefix=None):
    if torch.cuda.is_available():
        COMMAND = "nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv"
        process = subprocess.Popen(COMMAND.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()

        pre = ""
        if prefix is not None:
            pre = prefix
        if logger is not None:
            logger.debug(f'{pre}: {output.decode()}')
        else:
            print(f'{pre}: {output.decode()}')


@timing_decorator
def cluster_graphs(data, edge_scores, eps: float = 0.35, verbose=False):
    """
    Cluster the graphs using DBSCAN
    :param eps: eps distance for DBSCAN
    :param data: batch data from pyg
    :param edge_scores: predicted value without logit from GNN
    :param verbose:
    :return:
    """
    logger = logging.getLogger(__name__)

    device = edge_scores.device  # assuming edge_scores is your model output, it should be on the correct device
    data = data.to(device)

    # DBSCAN for graphs
    edge_batch_id = data.batch[data.edge_index[0]]
    assert edge_batch_id.shape == edge_scores.shape, f"[]{edge_batch_id.shape} != {edge_scores.shape}"

    # Apply the sigmoid function
    edge_scores_logit = 1 - torch.sigmoid(edge_scores)

    num_tracks = []
    # num_tracks = torch.zeros(data.num_graphs, dtype=torch.float16, requires_grad=True)
    for gr_id in range(data.num_graphs):

        num_nodes = data[gr_id].num_nodes
        if num_nodes == 0 or data[gr_id].edge_index.shape[0] != 2 or data[gr_id].edge_index.shape[1] < 1:
            logger.debug(
                f"The graph (evt: {data[gr_id].evt_num.item()}, "
                f"run:  {data[gr_id].run_num.item()}) has no nodes or edges."
            )
            continue  # Skip this iteration

        if not torch.any(edge_batch_id == gr_id):
            continue  # Skip this iteration

        dd = Data(edge_index=data[gr_id].edge_index)
        # Get the number of nodes
        num_nodes = data[gr_id].num_nodes
        # Initialize an empty adjacency matrix
        adj_matrix = torch.ones((num_nodes, num_nodes), device=device)
        # Fill in the adjacency matrix using edge_index
        adj_matrix[dd.edge_index[0], dd.edge_index[1]] = edge_scores_logit[edge_batch_id == gr_id]
        adj_matrix[dd.edge_index[1], dd.edge_index[0]] = edge_scores_logit[edge_batch_id == gr_id]

        # Run DBSCAN on the adjacency matrix
        cluster_labels = DBSCAN(eps=eps, min_samples=2, metric='precomputed').fit_predict(
            adj_matrix.detach().cpu().numpy())

        # Get the cluster labels
        num_tracks.append(len(np.unique(cluster_labels[cluster_labels >= 0])))

    # return torch.tensor(num_tracks, dtype=torch.int32)
    return np.array(num_tracks)
