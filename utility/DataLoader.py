# System imports
import os

# External imports
import numpy as np
import pandas as pd
import uproot as up
import torch
from torch.utils.data import Dataset, random_split
import torch_geometric

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataloader import default_collate


def get_data_loaders(name, batch_size, distributed=False,
                     n_workers=0, rank=None, n_ranks=None,
                     **data_args):
    """This may replace the datasets function above"""
    collate_fn = default_collate
    if name == 'test':
        train_dataset, valid_dataset = get_datasets(**data_args)
    else:
        raise Exception('Dataset %s unknown' % name)

    # Construct the data loaders
    loader_args = dict(batch_size=batch_size, collate_fn=collate_fn,
                       num_workers=n_workers)
    train_sampler, valid_sampler = None, None
    if distributed:
        train_sampler = DistributedSampler(train_dataset, rank=rank, num_replicas=n_ranks)
        valid_sampler = DistributedSampler(valid_dataset, rank=rank, num_replicas=n_ranks)
    train_data_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        **loader_args
    )
    valid_data_loader = (
        DataLoader(
            valid_dataset,
            sampler=valid_sampler,
            **loader_args
        )
        if valid_dataset is not None else None
    )
    return train_data_loader, valid_data_loader


def load_graph(filename):
    with np.load(filename) as f:
        x, y = f['X'], f['y']
        Ri_rows, Ri_cols = f['Ri_rows'], f['Ri_cols']
        Ro_rows, Ro_cols = f['Ro_rows'], f['Ro_cols']
        n_edges = Ri_cols.shape[0]
        edge_index = np.zeros((2, n_edges), dtype=int)
        edge_index[0, Ro_cols] = Ro_rows
        edge_index[1, Ri_cols] = Ri_rows
    return x, edge_index, y


class GNNTrackData(Dataset):
    """PyTorch dataset specification for hit graphs"""

    def __init__(self, root_path, collection, tree_name='dp'):
        with up.open(root_path) as f:
            node = f[tree_name].arrays([f'{collection}_{i}' for i in ["x", "y", "z"]])
            edge = f[tree_name].arrays([f'{collection}_{i}' for i in ["start", "end"]])
            truth = f[tree_name].arrays([f'{collection}_{i}' for i in ["truth"]])

            self.data = {
                'node': node,
                'edge': edge,
                'truth': truth,
            }

    def __getitem__(self, index):
        x, edge_index, y = load_graph(self.filenames[index])
        # Compute weights
        w = y * self.real_weight + (1 - y) * self.fake_weight
        return torch_geometric.data.Data(
            x=torch.from_numpy(x),
            edge_index=torch.from_numpy(edge_index),
            y=torch.from_numpy(y), w=torch.from_numpy(w),
            i=index
        )

    def __len__(self):
        return len(self.filenames)


def get_datasets(n_train, n_valid, input_dir=None, filelist=None, real_weight=1.0):
    data = GNNTrackData(
        input_dir=input_dir,
        filelist=filelist,
        n_samples=n_train + n_valid,
        real_weight=real_weight
    )
    # Split into train and validation
    train_data, valid_data = random_split(data, [n_train, n_valid])
    return train_data, valid_data
