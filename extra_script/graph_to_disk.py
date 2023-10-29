# External imports
from collections import defaultdict

import numpy as np
import pandas as pd
import uproot as up
import torch
import torch_geometric
import os

from typing import Generator, List

import plotly.graph_objects as go
import plotly.express as px


def load_ntuples(
        f_path, tree_name, branch_name, col, chunk_size="100 MB", momentum_predict=True, e0=8000, scale_b=1,
        graph_with_bfield=True, only_bfield_y=False,
        scale_r = 0.1, scale_theta = 10,
) -> Generator[List[torch_geometric.data.Data], None, None]:
    def convert_to_graph(ch) -> list[torch_geometric.data.Data]:
        g_data = []
        for index, eve in enumerate(ch):
            bfield = []
            if graph_with_bfield:
                bfield = [
                    eve[f'{col}_Bx'].to_numpy().reshape(-1, 1) * scale_b,
                    eve[f'{col}_By'].to_numpy().reshape(-1, 1) * scale_b,
                    eve[f'{col}_Bz'].to_numpy().reshape(-1, 1) * scale_b,
                ] if not only_bfield_y else [
                    eve[f'{col}_By'].to_numpy().reshape(-1, 1) * scale_b,
                ]

            node = np.hstack([*[
                eve[f'{col}_x'].to_numpy().reshape(-1, 1),
                eve[f'{col}_y'].to_numpy().reshape(-1, 1),
                eve[f'{col}_z'].to_numpy().reshape(-1, 1)
            ], *bfield])

            edge_index = np.hstack([
                eve[f'{col}_start'].to_numpy().reshape(-1, 1),
                eve[f'{col}_end'].to_numpy().reshape(-1, 1),
            ]).transpose()

            edge_features = np.hstack([
                eve[f'{col}_distance'].to_numpy().reshape(-1, 1) * scale_r,
                eve[f'{col}_Theta'].to_numpy().reshape(-1, 1) * scale_theta,
                eve[f'{col}_Phi'].to_numpy().reshape(-1, 1)
            ])

            y = eve[f'{col}_truth'].to_numpy()
            truth_w = eve[f'{col}_weight']
            truth_num_track = eve[f'{col}_global_num_tracks']
            # re-weight truth edge with fake one
            w = y * (1 - truth_w) / truth_w + (1 - y) * (1 - truth_w)

            graph = torch_geometric.data.Data(
                x=torch.from_numpy(node.astype(np.float32)),
                edge_index=torch.from_numpy(edge_index.astype(np.int64)),
                y=torch.from_numpy(y.astype(np.float32)),
                w=torch.from_numpy(w.astype(np.float32)),
                n=torch.from_numpy(np.array([truth_num_track]).astype(np.int64)),
                i=torch.from_numpy(np.array([report.start + index])),
                run_num=torch.from_numpy(np.array([eve['run_num']])),
                evt_num=torch.from_numpy(np.array([eve['evt_num']])),
                edge_attr=torch.from_numpy(edge_features.astype(np.float32)),
                truth_w=torch.from_numpy(np.array([truth_w]).astype(np.float32)),
            )
            if momentum_predict:
                p = eve[f'{col}_p'].to_numpy()
                graph.p = torch.from_numpy(p.astype(np.float32)) / e0

            g_data.append(graph)
        return g_data

    for chunk, report in up.iterate(
            [{f_path: tree_name}],
            step_size=chunk_size,
            filter_name=branch_name,
            cut=f'{col}_weight>0',
            report=True,
    ):
        print(f'Loading {report.start} to {report.stop}...', flush=True)
        data = convert_to_graph(chunk)
        yield data


def graph_summary(graphs: list[torch_geometric.data.Data]) -> pd.DataFrame:
    def categorize_z(graphs):
        results = defaultdict(list)

        # Define your layers (converted to millimeters and then back to original units)
        tag_layers = torch.tensor([-300, -200, -100, 0, 100, 200, 300]) - 307.78
        rec_layers = torch.tensor([-86.2500, -71.2500, -55.2500, -40.2500, -4.2500, 86.2500]) + 94.03

        # Concatenate the layers
        layers = torch.cat([tag_layers, rec_layers]).view(1, -1)  # Reshape to (1, N)

        # The error margin (converted to original units)
        margin = 5  # Convert to original units

        # Iterate over the graphs
        for data in graphs:
            z_values = data.x[:, 2].view(-1, 1)  # Get the z-values and reshape to (M, 1)

            # Subtract the z_values from layers, take the absolute value, and compare with margin
            # This gives a tensor of shape (M, N) with True where the z-value is close to a layer
            categories = (z_values - layers).abs() <= margin

            # Assign the categories
            _, z_categories = categories.max(dim=1)  # This gives the index of the True value in each row

            # Get the unique categories and their counts
            unique_categories, counts = torch.unique(z_categories, return_counts=True)

            # Count edges connected to each layer
            for _, layer in enumerate(range(len(layers[0]))):
                # Nodes in the current layer
                nodes_in_layer = torch.where(z_categories == layer)[0]

                nodes_in_layer_np = nodes_in_layer.cpu().numpy()
                edge_index_0_np = data.edge_index[0].cpu().numpy()
                edge_index_1_np = data.edge_index[1].cpu().numpy()
                edges_in_layer_mask = \
                    np.isin(edge_index_0_np, nodes_in_layer_np) | np.isin(edge_index_1_np, nodes_in_layer_np)

                # Edges in the current layer
                edges_in_layer = edges_in_layer_mask.sum()

                # True edges in the current layer
                true_edges_in_layer = (torch.tensor(edges_in_layer_mask, dtype=torch.int) & (data.y == 1)).sum()

                # Add counts to results
                if max(unique_categories) < 7:
                    if layer >= 7: break
                else:
                    if layer < 7: continue

                idx = layer if max(unique_categories) < 7 else (layer - 7)
                layer_item = f'tag_{idx}' if max(unique_categories) < 7 else f'rec_{idx}'

                if layer in unique_categories:
                    idx = torch.where(unique_categories == layer)[0]

                results[f'Layer_{layer_item}_num_node'].append(counts[idx].item())
                results[f'Layer_{layer_item}_num_edge'].append(edges_in_layer.item())
                results[f'Layer_{layer_item}_num_truth_edge'].append(true_edges_in_layer.item())

        return results

    re_df = pd.DataFrame({
        'num_nodes': [data.num_nodes for data in graphs],
        'num_edges': [data.num_edges for data in graphs],
        'num_true_edges': [data.y.sum().item() for data in graphs],
        **categorize_z(graphs),
    })

    return re_df


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Convert ROOT to torch_geometric graph, save to disk")
    # change args to only one file
    parser.add_argument('file', type=str, help="the input root file")
    parser.add_argument('-o', '--output', type=str, default='output', help="the output directory")
    parser.add_argument('-c', '--chunk', type=str, default='50 MB', help="the chunk size")
    parser.add_argument('-m', '--momentum_predict', type=bool, default=False, help="whether to predict momentum")
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

    origin_br = ["x", "y", "z", "start", "end", "weight", "truth", "global_num_tracks", "distance", "Theta", "Phi"]
    if args.momentum_predict: origin_br += ["p"]
    if args.bfield: origin_br += ["Bx", "By", "Bz"] if not args.only_bfield_y else ["By"]
    # load data
    file_path = args.file
    print(f"Processing {file_path}", flush=True)
    cur_tree = up.open(f'{file_path}:dp')
    # Get branch names
    branches = cur_tree.keys()

    collections = set(branch.rsplit('_', 1)[0] for branch in branches if branch.rsplit('_', 1)[-1] in origin_br)
    print(collections, flush=True)

    for col in list(collections):
        os.makedirs(os.path.join(args.output, col), exist_ok=True)
        print(f"[{col}]:  Output to {os.path.join(args.output, col)}", flush=True)

        collection_branch = [f'{col}_{br}' for br in origin_br]
        collection_branch += ['evt_num', 'run_num']
        graph_data = load_ntuples(
            file_path, 'dp', collection_branch, col, chunk_size=args.chunk,
            momentum_predict=args.momentum_predict, e0=args.e0,
            scale_b=args.scale_b, graph_with_bfield=args.bfield,
            only_bfield_y=args.only_bfield_y,
            scale_r=args.scale_r, scale_theta=args.scale_theta,
        )

        df_col = []
        n_graph = 0
        while True:
            try:
                data_list = next(graph_data)
                print(f"[{col}]:  Saving {n_graph}th graph", flush=True)
                torch.save(data_list, os.path.join(args.output, col, f'graph_list.{n_graph}.{args.tag}.pt'))

                # df_col.append(graph_summary(data_list))
                n_graph += 1
            except StopIteration:
                break

        # df = pd.concat(df_col, ignore_index=True)
        # os.makedirs(os.path.join(args.output, 'stats'), exist_ok=True)
        # df.to_csv(os.path.join(args.output, 'stats', f'{col}.{args.tag}.csv'), index=False)
