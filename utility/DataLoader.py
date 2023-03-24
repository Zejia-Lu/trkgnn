# System imports
import os
from typing import Union, List, Tuple

# External imports
import numpy as np
import pandas as pd
import uproot as up
import torch
from torch_geometric.loader import DataLoader
import torch_geometric
from torch_geometric.data import Dataset, Data

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataloader import default_collate
from uproot.models.TTree import Model_TTree_NumEntries
from sklearn.model_selection import train_test_split

from utility.Control import cfg
from utility.FunctionTime import timing_decorator


@timing_decorator
def get_data_loaders(
        input_dir, chunk_size, batch_size,
        distributed=False, n_workers=0, rank=None, n_ranks=None
):
    # load chunk
    graph_branch = [f'{cfg["data"]["collection"]}_{i}' for i in ["x", "y", "z", "start", "end", "truth", "weight"]]
    chunk_generator = load_ntuples(
        input_dir, cfg['data']['tree_name'], graph_branch, cfg["data"]["collection"], chunk_size
    )

    while True:
        try:
            chunk_data = next(chunk_generator)

            train_data, test_data = train_test_split(chunk_data, test_size=0.3, random_state=cfg['rndm'])
            train_dataset = GNNTrackData(train_data)
            valid_dataset = GNNTrackData(test_data)

            collate_fn = default_collate
            loader_args = dict(
                batch_size=batch_size,
                collate_fn=collate_fn,
                num_workers=n_workers,
                pin_memory=True,
            )

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

            print(f"Dataset size: {len(train_dataset)}")
            print(f"Total number of GPUs: {n_ranks}")
            print(f"Rank (GPU index): {rank}")

            sample_indices = [i for i in train_data_loader]
            print(f"[ {rank} ] Number of samples assigned to GPU {rank}: {len(sample_indices)}")
            print(f"[ {rank} ] Assigned sample indices for GPU {rank}: {sample_indices}")

            sample_indices = [i for i in valid_data_loader]
            print(f"[ {rank} ] Number of samples assigned to GPU {rank}: {len(sample_indices)}")
            print(f"[ {rank} ] Assigned sample indices for GPU {rank}: {sample_indices}")

            yield train_data_loader, valid_data_loader

        except StopIteration:
            print("All chunks are loaded")
            break

    pass


@timing_decorator
def get_entries(file_path, tree_name):
    return np.sum([
        up.open(f"{path[0]}:{path[1]}", custom_classes={"TTree": Model_TTree_NumEntries})
        .all_members["fEntries"][0]
        for path in up._util.regularize_files(f"{file_path}:{tree_name}")
    ])


@timing_decorator
def load_ntuples(file_path, tree_name, branch_name, col, chunk_size="100 MB"):
    for chunk, report in up.iterate(
            [{file_path: tree_name}],
            step_size=chunk_size,
            filter_name=branch_name,
            report=True
    ):
        print(report)
        process_len = report.stop - report.start
        data = []
        for index, eve in enumerate(chunk):
            node = np.hstack([
                eve[f'{col}_x'].to_numpy().reshape(-1, 1),
                eve[f'{col}_y'].to_numpy().reshape(-1, 1),
                eve[f'{col}_z'].to_numpy().reshape(-1, 1)
            ])
            edge_index = np.hstack([
                eve[f'{col}_start'].to_numpy().reshape(-1, 1),
                eve[f'{col}_end'].to_numpy().reshape(-1, 1),
            ]).transpose()
            y = eve[f'{col}_truth'].to_numpy()  # .squeeze()
            truth_w = eve[f'{col}_weight']
            # re-weight truth edge with fake one
            w = y * (1 - truth_w) / truth_w + (1 - y) * (1 - truth_w)

            idx = torch.from_numpy(np.array([report.start + index]))
            graph = torch_geometric.data.Data(
                x=torch.from_numpy(node.astype(np.float32)),
                edge_index=torch.from_numpy(edge_index.astype(np.int64)),
                y=torch.from_numpy(y.astype(np.float32)),
                w=torch.from_numpy(w.astype(np.float32)),
                i=idx
            )
            data.append(graph)
        print('data -> ', data)
        yield data


class GNNTrackData(Dataset):
    """PyTorch dataset specification for hit graphs"""

    @timing_decorator
    def __init__(self, data):
        super().__init__()

        self.total_len = len(data)
        self.data = data

        # self.data_gen = load_ntuples(input_dir, tree_name, graph_branch, col, chunk_size="100 MB")

    def len(self) -> int:
        return self.total_len

    def get(self, idx: int) -> Data:
        return self.data[idx]


if __name__ == '__main__':
    from utility.Control import load_config
    from utility.FunctionTime import print_accumulated_times

    load_config('/Users/avencast/PycharmProjects/trkgnn/configs/mpnn.yaml')
    load_gen = get_data_loaders(cfg['data']['input_dir'], chunk_size=10, batch_size=2)

    while True:
        try:
            a, b = next(load_gen)
            print(len(a))
            print(len(b))
        except StopIteration:
            print("Finish")
            break

    print_accumulated_times()
