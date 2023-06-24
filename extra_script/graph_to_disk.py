# External imports
import numpy as np
import uproot as up
import torch
import torch_geometric
import os


def load_ntuples(f_path, tree_name, branch_name, col, chunk_size="100 MB", momentum_predict=True, e0=8000):
    def convert_to_graph(ch):
        g_data = []
        for index, eve in enumerate(ch):
            node = np.hstack([
                eve[f'{col}_x'].to_numpy().reshape(-1, 1),
                eve[f'{col}_y'].to_numpy().reshape(-1, 1),
                eve[f'{col}_z'].to_numpy().reshape(-1, 1)
            ])
            edge_index = np.hstack([
                eve[f'{col}_start'].to_numpy().reshape(-1, 1),
                eve[f'{col}_end'].to_numpy().reshape(-1, 1),
            ]).transpose()
            y = eve[f'{col}_truth'].to_numpy()
            truth_w = eve[f'{col}_weight']
            # re-weight truth edge with fake one
            w = y * (1 - truth_w) / truth_w + (1 - y) * (1 - truth_w)

            graph = torch_geometric.data.Data(
                x=torch.from_numpy(node.astype(np.float32)),
                edge_index=torch.from_numpy(edge_index.astype(np.int64)),
                y=torch.from_numpy(y.astype(np.float32)),
                w=torch.from_numpy(w.astype(np.float32)),
                i=torch.from_numpy(np.array([report.start + index])),
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
        data = convert_to_graph(chunk)
        yield data


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Convert ROOT to torch_geometric graph, save to disk")
    # change args to only one file
    parser.add_argument('file', type=str, help="the input root file")
    parser.add_argument('-o', '--output', type=str, default='output', help="the output directory")
    parser.add_argument('-c', '--chunk', type=str, default='50 MB', help="the chunk size")
    parser.add_argument('-m', '--momentum_predict', type=bool, default=True, help="whether to predict momentum")
    parser.add_argument('-e', '--e0', type=float, default=8000, help="the beam energy")
    parser.add_argument('-t', '--tag', type=str, default='out', help="the output file name suffix")

    args = parser.parse_args()

    origin_br = ["x", "y", "z", "start", "end", "weight", "truth"]
    if args.momentum_predict: origin_br += ["p"]
    # load data
    file_path = args.file
    print(f"Processing {file_path}")
    cur_tree = up.open(f'{file_path}:dp')
    # Get branch names
    branches = cur_tree.keys()

    collections = set(branch.rsplit('_', 1)[0] for branch in branches if branch.rsplit('_', 1)[-1] in origin_br)
    print(collections)

    for col in list(collections):
        os.makedirs(os.path.join(args.output, col), exist_ok=True)
        print(f"[{col}]:  Output to {os.path.join(args.output, col)}")

        collection_branch = [f'{col}_{br}' for br in origin_br]

        graph_data = load_ntuples(
            file_path, 'dp', collection_branch, col, chunk_size=args.chunk,
            momentum_predict=args.momentum_predict, e0=args.e0
        )

        n_graph = 0
        while True:
            try:
                data_list = next(graph_data)
                print(f"[{col}]:  Saving {n_graph}th graph")
                torch.save(data_list, os.path.join(args.output, col, f'graph_list.{n_graph}.{args.tag}.pt'))
                n_graph += 1
            except StopIteration:
                break

        # for index, eve in enumerate(data):
        #     torch.save(eve, os.path.join(args.output, f'graph_{index}.pt'))
