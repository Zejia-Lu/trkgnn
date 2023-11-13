# External imports
from collections import defaultdict

import numpy as np
import pandas as pd
import uproot as up
import torch
import torch_geometric
import os

from typing import Generator, List


def load_ntuples(
        f_path, tree_name, branch_name, chunk_size="100 MB", e0=8000, scale_b=1,
        graph_with_bfield=True, only_bfield_y=False,
        scale_r=0.1, scale_theta=10,
) -> Generator[List[torch_geometric.data.Data], None, None]:
    def convert_to_graph(ch) -> list[torch_geometric.data.Data]:
        g_data = []
        for index, eve in enumerate(ch):
            bfield = []
            if graph_with_bfield:
                bfield = [
                    eve[f'Bx'].to_numpy().reshape(-1, 1) * scale_b,
                    eve[f'By'].to_numpy().reshape(-1, 1) * scale_b,
                    eve[f'Bz'].to_numpy().reshape(-1, 1) * scale_b,
                ] if not only_bfield_y else [
                    eve[f'By'].to_numpy().reshape(-1, 1) * scale_b,
                ]

            node = np.hstack([*[
                eve[f'x'].to_numpy().reshape(-1, 1),
                eve[f'y'].to_numpy().reshape(-1, 1),
                eve[f'z'].to_numpy().reshape(-1, 1)
            ], *bfield])

            edge_index = np.hstack([
                eve[f'start'].to_numpy().reshape(-1, 1),
                eve[f'end'].to_numpy().reshape(-1, 1),
            ]).transpose()

            edge_features = np.hstack([
                eve[f'distance'].to_numpy().reshape(-1, 1) * scale_r,
                eve[f'Theta'].to_numpy().reshape(-1, 1) * scale_theta,
                eve[f'Phi'].to_numpy().reshape(-1, 1),
                eve[f'p'].to_numpy().reshape(-1, 1) / e0,
            ])

            y = eve[f'has']
            z = eve[f'Vz']

            graph = torch_geometric.data.Data(
                x=torch.from_numpy(node.astype(np.float32)),
                edge_index=torch.from_numpy(edge_index.astype(np.int64)),
                y=torch.from_numpy(np.array([y]).astype(np.float32)),
                z=torch.from_numpy(np.array([z]).astype(np.float32)),
                i=torch.from_numpy(np.array([report.start + index])),
                edge_attr=torch.from_numpy(edge_features.astype(np.float32)),
            )

            g_data.append(graph)
        return g_data

    for chunk, report in up.iterate(
            [{f_path: tree_name}],
            step_size=chunk_size,
            filter_name=branch_name,
            report=True,
    ):
        print(f'Loading {report.start} to {report.stop}...', flush=True)
        data = convert_to_graph(chunk)
        yield data


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Convert ROOT to torch_geometric graph, save to disk")
    # change args to only one file
    parser.add_argument('file', type=str, help="the input root file")
    parser.add_argument('-o', '--output', type=str, default='output', help="the output directory")
    parser.add_argument('-c', '--chunk', type=str, default='50 MB', help="the chunk size")
    parser.add_argument('-e', '--e0', type=float, default=8000, help="the beam energy")
    parser.add_argument('-t', '--tag', type=str, default='out', help="the output file name suffix")
    parser.add_argument('-b', '--bfield', action='store_true', default=False, help="whether to use bfield")
    parser.add_argument('-s', '--scale_b', type=float, default=100,
                        help="the scale factor for magnetic field (general ~ 1.5T, so default is 100)")
    parser.add_argument('-y', '--only_bfield_y', action='store_true', default=False,
                        help="whether to use only bfield_y")
    parser.add_argument('--scale_r', type=float, default=0.1,
                        help="the scale factor for distance (general ~ 10 to 100, so default is 0.1)")
    parser.add_argument('--scale_theta', type=float, default=10,
                        help="the scale factor for theta radian (general ~ 0.1 rad, so default is 10)")

    args = parser.parse_args()

    col = "vertex"
    origin_br = ["x", "y", "z", "start", "end", "p", "distance", "Theta", "Phi", "has", "Vz"]
    if args.bfield: origin_br += ["Bx", "By", "Bz"] if not args.only_bfield_y else ["By"]
    # load data
    file_path = args.file
    print(f"Processing {file_path}", flush=True)
    cur_tree = up.open(f'{file_path}:{col}')
    # Get branch names
    branches = cur_tree.keys()

    os.makedirs(os.path.join(args.output, col), exist_ok=True)
    print(f"[{col}]:  Output to {os.path.join(args.output, col)}", flush=True)
    graph_data = load_ntuples(
        file_path, col, origin_br, chunk_size=args.chunk,
        e0=args.e0,
        scale_b=args.scale_b, graph_with_bfield=args.bfield,
        only_bfield_y=args.only_bfield_y,
        scale_r=args.scale_r, scale_theta=args.scale_theta,
    )

    df_col = []
    n_graph = 0
    while True:
        try:
            data_list = next(graph_data)
            print(f"Saving {n_graph}th graph", flush=True)
            torch.save(data_list, os.path.join(args.output, col, f'graph_list.{n_graph}.{args.tag}.pt'))

            # df_col.append(graph_summary(data_list))
            n_graph += 1
        except StopIteration:
            break
