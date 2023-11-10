import gc
import logging
import pickle
import os
import torch
import torch_geometric
import torch_geometric.utils as pyg_utils
from torch_geometric.utils import subgraph
import numpy as np
from collections import deque, defaultdict
from copy import deepcopy

from utility.Control import cfg
from utility.DataLoader import get_data_loaders
from utility.EverythingNeeded import config_logging, build_model
from utility.FunctionTime import timing_decorator
from utility.DTrack import DTrack

from pyvis.network import Network


def setup():
    # check OS
    # Check if CUDA is available
    if torch.cuda.is_available():
        print("CUDA is available.")

        # Check the number of available GPUs
        num_gpus = torch.cuda.device_count()
        print(f"Number of available GPUs: {num_gpus}, but only 1 is used.")

        device = torch.device("cuda:0")
    # elif torch.has_mps:
    #     print("mps is available.")
    #     device = torch.device("mps")
    else:
        print("CUDA is not available.")
        device = torch.device("cpu")

    cfg['device'] = device
    # cfg['data']['read_from_graph'] = False
    cfg['momentum_predict'] = True

    # Run all events
    if 'global_stop' in cfg['data']:
        del cfg['data']['global_stop']


@timing_decorator
@torch.no_grad()
def predict(input_dir: list[str], model_dir: str, output_dir: str, truth: bool = False):
    config_logging(True, output_dir=output_dir, prefix='apply')
    setup()
    logger = logging.getLogger("Apply.Momentum")

    logger.info(f"input_dir: {input_dir}")
    logger.info(f"model_dir: {model_dir}")
    logger.info(f"output_dir: {output_dir}")
    if truth: logger.info(f"Truth Mode !")

    # Build model and load model
    model = build_model(cfg['device'], distributed=False)
    tar = torch.load(model_dir, map_location=cfg['device'])
    model.load_state_dict(tar['model'])
    model.to(cfg['device'])
    model.eval()

    # Build data loader
    for i, data_dir in enumerate(input_dir):
        logger.info(f"Processing {i + 1}/{len(input_dir)}: {data_dir}")

        data_generator = get_data_loaders(
            data_dir,
            chunk_size=cfg['data']['chunk_size'],
            batch_size=cfg['data']['batch_size'],
            distributed=False,
            n_workers=cfg['data']['n_workers'],
            shuffle=True,
            apply=True,
        )

        output_graph_dir = os.path.join(output_dir, cfg['data']['collection'])
        os.makedirs(output_graph_dir, exist_ok=True)

        itr = 0
        while True:
            try:
                apply_loader = next(data_generator)
                logger.info(f"Processing {itr + 1}th iteration with {len(apply_loader)} batches.")

                predicted_graph_list = []
                analyzed_tracks_list = []
                num_graphs = 0
                # Loop over batches
                for j, batch in enumerate(apply_loader):
                    batch = batch.to(cfg['device'])
                    p_pred_all, y_pred = model(batch)
                    # y_pred = torch.sigmoid(new_y_score)
                    batch.edge_attr = torch.cat([batch.edge_attr, y_pred.unsqueeze(-1), p_pred_all.unsqueeze(-1)],
                                                dim=1)
                    batch = batch.to("cpu")

                    if 'tracking_layers' in cfg['data']:
                        num_batches = 0
                        for idx, _ in enumerate(batch):
                            paths = cluster(batch[idx], threshold=cfg['data']['threshold'])
                            analyzed_tracks_list.append(analyze_tracks(batch[idx], paths))
                            num_batches += 1
                        logger.info(f"Number of graphs: {num_batches} processed in {j}th batch.")

                    predicted_graph_list += batch.to_data_list()
                    num_graphs += batch.num_graphs

                torch.save(predicted_graph_list, os.path.join(output_graph_dir, f"momentum_{itr}.pt"))
                torch.save(analyzed_tracks_list, os.path.join(output_graph_dir, f"tracks_{itr}.lt"))

                logger.info(f"Number of graphs: {num_graphs} processed in {itr + 1}th iteration.")
                itr += 1
            except StopIteration:
                print("Finish")
                # logout to root file
                logger.info(f"Finish predicting file: {data_dir}")
                break


@timing_decorator
def cluster(gr: torch_geometric.data.Data, threshold: float = 0.5):
    logger = logging.getLogger("Apply.Momentum.Cluster")

    @timing_decorator
    def classify_layers(z_values, layer_centers, tolerance):
        # Calculate the absolute difference between z-values and layer centers
        differences = np.abs(z_values - np.array(layer_centers).reshape(1, -1))
        # Find the index of the minimum difference
        layer_indices = np.argmin(differences, axis=1)
        # Determine if the minimum difference is within the tolerance, otherwise set to -1 or some null value
        layers = np.where(np.min(differences, axis=1) <= tolerance, layer_indices, -1)

        return layers

    layer = classify_layers(
        z_values=gr.x[:, 2].numpy().reshape(-1, 1),
        layer_centers=cfg['data']['tracking_layers'],
        tolerance=1.0
    )

    gr.x = torch.cat([gr.x, torch.tensor(layer).unsqueeze(-1)], dim=-1)

    mask = gr.edge_attr[:, -2] > threshold

    # Create the subgraph
    sub_data = torch_geometric.data.Data(
        x=gr.x,
        edge_index=gr.edge_index[:, mask],
        edge_attr=gr.edge_attr[mask, :],
    )

    out_degrees = pyg_utils.degree(sub_data.edge_index[0], sub_data.num_nodes)
    in_degrees = pyg_utils.degree(sub_data.edge_index[1], sub_data.num_nodes)

    # find initial start nodes with only out-edges and no in-edges
    start_nodes = (out_degrees > 0) & (in_degrees == 0)
    start_nodes = start_nodes.nonzero(as_tuple=False).view(-1).numpy()

    @timing_decorator
    def bfs_for_multiple_starts_with_layers(data, start_nodes_, layers):
        paths_from_start = defaultdict(list)
        queue = deque([(start_node, [start_node]) for start_node in start_nodes_])  # (current_node, path)

        edge_index = data.edge_index.cpu().numpy()

        # Create a children dictionary to avoid recomputing it
        children = defaultdict(list)
        for start, end in edge_index.T:
            children[start].append(end)

        while queue:
            current_node, path = queue.popleft()
            current_layer = layers[current_node].item()
            for child in children[current_node]:
                child_layer = layers[child].item()
                # Only add the child if it's in the next layer
                if child_layer == current_layer + 1:

                    index_to_modify = next(
                        (i for i, sublist in enumerate(paths_from_start[path[0]]) if sublist == path)
                        , None
                    )

                    if index_to_modify is not None:
                        paths_from_start[path[0]][index_to_modify].append(child)
                    else:
                        index_to_modify = len(paths_from_start[path[0]])
                        paths_from_start[path[0]].append([*path, child])
                    queue.append((child, deepcopy(paths_from_start[path[0]][index_to_modify])))

        return paths_from_start

    layers = sub_data.x[:, -1].long()
    paths_from_start = bfs_for_multiple_starts_with_layers(sub_data, start_nodes, layers)

    # visualize graph
    # net = Network()
    # net.add_nodes(range(sub_data.num_nodes))
    # for source, to, value, width in zip(sub_data.edge_index[0].numpy(), sub_data.edge_index[1].numpy(),
    #                                     sub_data.edge_attr[:, -2].numpy(), sub_data.edge_attr[:, -1].numpy()):
    #     net.add_edge(int(source), int(to), width=float(value), physics=False)
    #
    # net.show('test.html', notebook=False)

    return paths_from_start


@timing_decorator
def analyze_tracks(graph: torch_geometric.data.Data, paths: dict[list]):
    logger = logging.getLogger("Apply.Momentum.Analyze")

    trajectories = []
    for start_points in paths.keys():
        for path in paths[start_points]:
            traj = DTrack()
            _, sub_edges = subgraph(path, graph.edge_index, graph.edge_attr)

            traj.run_num = graph.run_num.item()
            traj.evt_num = graph.evt_num.item()

            traj.no_hits = len(path)
            traj.p_i = sub_edges[0, -1].item()
            traj.p_f = sub_edges[-1, -1].item()
            traj.p_avg = np.mean(sub_edges[:, -1].numpy())

            traj.vertex_hit = graph.x[path[0], :3].numpy()
            traj.end_hit = graph.x[path[-1], :3].numpy()

            trajectories.append(traj)

    return trajectories
