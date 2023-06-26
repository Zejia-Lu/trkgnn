import gc
import logging
import warnings
import os
from typing import Tuple
from collections import defaultdict

import pandas as pd
import torch
import torch_geometric
import uproot as up
import numpy as np

# import community as community_louvain
import networkx as nx
from sklearn.cluster import DBSCAN

from utility.Control import cfg
from utility.DataLoader import get_data_loaders
from utility.EverythingNeeded import config_logging, build_model
from utility.FunctionTime import timing_decorator
from utility.DTrack import DTrack


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
    cfg['data']['read_from_graph'] = False

    # Run all events
    del cfg['data']['global_stop']


@timing_decorator
@torch.no_grad()
def apply_to_ds(input_dir: list[str], model_dir: str, output_dir: str):
    config_logging(True, output_dir=output_dir, prefix='apply')
    setup()
    logger = logging.getLogger("Apply")

    logger.info(f"input_dir: {input_dir}")
    logger.info(f"model_dir: {model_dir}")
    logger.info(f"output_dir: {output_dir}")

    # Build model and load model
    model = build_model(cfg['device'], distributed=False)
    tar = torch.load(model_dir, map_location=cfg['device'])
    model.load_state_dict(tar['model'])
    model.to(cfg['device'])
    model.eval()

    # For each input_dir, apply the model
    df_edge_list = []
    df_node_list = []
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

        # setup output root file
        write_branch_dicts = {
            "evt_num": "i",
            "run_num": "i",
            "no_hits": "i",
            "p_i": "f",
            "p_f": "f",
            "vertex_x": "f",
            "vertex_y": "f",
            "vertex_z": "f",
            "end_x": "f",
            "end_y": "f",
            "end_z": "f",
        }
        output_root_dir = os.path.join(output_dir, f"out_roots")
        os.makedirs(output_root_dir, exist_ok=True)
        output_root_file = up.recreate(os.path.join(output_root_dir, f"out_{i}.root"))
        output_root_file.mktree("gnn_tracks", write_branch_dicts, title="GNN Tracks")

        itr = 0
        while True:
            try:
                apply_loader = next(data_generator)
                # final tracks
                tracks = []
                # Loop over batches
                for j, batch in enumerate(apply_loader):
                    batch = batch.to(cfg['device'])
                    batch_out = model(batch)
                    if cfg['momentum_predict']:
                        y_pred, p_pred = batch_out
                    else:
                        y_pred = batch_out
                        p_truth, p_pred = None, None

                    y_pred = torch.sigmoid(y_pred)

                    # Unbatch the graphs
                    df_edge, df_node, all_graphs = unbatch_graphs(batch, y_pred, p_pred)

                    df_edge_list.append(df_edge)
                    df_node_list.append(df_node)

                    # Cluster the tracks
                    tracks += [*cluster_tracks(all_graphs)]

                # save tracks to root file
                export_to_root(output_root_file['gnn_tracks'], tracks, write_branch_dicts)

                itr += 1
            except StopIteration:
                print("Finish")
                output_root_file.close()
                # logout to root file
                logger.info(f"Output root file: {output_root_file}")
                break

    df_edge = pd.concat(df_edge_list).reset_index(drop=True)
    df_node = pd.concat(df_node_list).reset_index(drop=True)
    del df_edge_list
    del df_node_list
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # Save the results
    graph_dir = os.path.join(output_dir, 'graphs')
    os.makedirs(graph_dir, exist_ok=True)
    df_edge.to_csv(os.path.join(graph_dir, 'edge.csv'), index=False, header=True)
    df_node.to_csv(os.path.join(graph_dir, 'node.csv'), index=False, header=True)

    del df_edge, df_node


@timing_decorator
def unbatch_graphs(batch, y_pred, p_pred=None) -> Tuple[pd.DataFrame, pd.DataFrame, list[nx.Graph]]:
    all_edge_dfs = []
    all_node_dfs = []
    all_graphs = []
    edge_index_list = torch_geometric.utils.unbatch_edge_index(batch.edge_index, batch.batch)
    edge_split_sizes = [ei.size(1) for ei in edge_index_list]
    node_split_sizes = torch_geometric.utils.degree(batch.batch, dtype=torch.long).tolist()
    y_truth_list = batch.y.split(edge_split_sizes, dim=0)
    y_pred_list = y_pred.split(edge_split_sizes, dim=0)
    x_list = batch.x.split(node_split_sizes, dim=0)

    p_truth_list = None
    p_pred_list = None
    if p_pred is not None:
        p_truth_list = batch.p.split(edge_split_sizes, dim=0)
        p_pred_list = p_pred.split(edge_split_sizes, dim=0)

    for i in range(len(edge_index_list)):
        edge_index = edge_index_list[i].T  # transpose to match previous format
        x = x_list[i]  # node features for current graph

        edge_data_dict = {
            'evt_num': batch.evt_num[i].item(),
            'run_num': batch.run_num[i].item(),
            'edge_start_index': edge_index[:, 0].tolist(),
            'edge_end_index': edge_index[:, 1].tolist(),
            'y_truth': y_truth_list[i].tolist(),
            'y_pred': y_pred_list[i].tolist(),
        }

        node_data_dict = {
            'evt_num': batch.evt_num[i].item(),
            'run_num': batch.run_num[i].item(),
            'x': x[:, 0].tolist(),
            'y': x[:, 1].tolist(),
            'z': x[:, 2].tolist(),
        }

        if p_pred_list is not None:
            edge_data_dict['p_truth'] = p_truth_list[i].tolist()
            edge_data_dict['p_pred'] = p_pred_list[i].tolist()

        edge_df = pd.DataFrame(edge_data_dict)
        node_df = pd.DataFrame(node_data_dict)

        all_edge_dfs.append(edge_df)
        all_node_dfs.append(node_df)

        # Create new Graph
        graph = nx.Graph()
        graph.graph['evt_num'] = batch.evt_num[i].item()
        graph.graph['run_num'] = batch.run_num[i].item()
        for node_index, (x, y, z) in enumerate(zip(
                node_data_dict['x'], node_data_dict['y'], node_data_dict['z']
        )):
            graph.add_node(node_index, x=x, y=y, z=z)

        for start, end, y_t, y_p in zip(
                edge_data_dict['edge_start_index'],
                edge_data_dict['edge_end_index'],
                edge_data_dict['y_truth'],
                edge_data_dict['y_pred'],
        ):
            graph.add_edge(start, end, y_truth=y_t, y_pred=y_p)

        if p_pred_list is not None:
            for start, end, p_t, p_p in zip(
                    edge_data_dict['edge_start_index'],
                    edge_data_dict['edge_end_index'],
                    edge_data_dict['p_truth'],
                    edge_data_dict['p_pred'],
            ):
                graph[start][end]['p_truth'] = p_t
                graph[start][end]['p_pred'] = p_p

        # make sure the graph is undirected
        graph = graph.to_undirected()
        all_graphs.append(graph)

    return pd.concat(all_edge_dfs).reset_index(drop=True), pd.concat(all_node_dfs).reset_index(), all_graphs


# track clustering
@timing_decorator
def cluster_tracks(graph: list[nx.Graph]) -> list[DTrack]:
    """
    Cluster tracks using DBSCAN method
    :param graph: input list of graphs
    :return: batch: [ event: [ track: DTrack ] ]
    """
    clustered_tracks = []
    for gr in graph:
        # apply DBSCAN
        tracks_event = []
        # Convert the graph's adjacency matrix to a dense format
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            adj_matrix = nx.adjacency_matrix(gr, weight='y_pred').toarray()

        # Convert edge scores to distances
        distance_matrix = 1 - adj_matrix

        # Apply DBSCAN
        db = DBSCAN(eps=0.5, min_samples=2, metric='precomputed')
        db.fit(distance_matrix)
        node_labels = db.labels_

        subgraphs = defaultdict(nx.Graph)
        # Iterate over the nodes and add them to the corresponding subgraph
        for node, label in zip(gr.nodes, node_labels):
            if label >= 0:
                subgraphs[label].add_node(node)

        # Add the edges to the sub-graphs
        for subgraph in subgraphs.values():
            track = DTrack()
            track.from_graph(gr.subgraph(subgraph.nodes), cfg['data']['E0'])
            clustered_tracks.append(track)

    return clustered_tracks


@timing_decorator
def export_to_root(tree: up.TTree, tracks: list[DTrack], br_dicts: dict):
    track_dict = {k: np.array([]) for k in br_dicts.keys()}
    for t in tracks:
        track_dict['evt_num'] = np.append(track_dict['evt_num'], t.evt_num)
        track_dict['run_num'] = np.append(track_dict['run_num'], t.run_num)
        track_dict['no_hits'] = np.append(track_dict['no_hits'], t.no_hits)
        track_dict['p_i'] = np.append(track_dict['p_i'], t.p_i)
        track_dict['p_f'] = np.append(track_dict['p_f'], t.p_f)
        track_dict['vertex_x'] = np.append(track_dict['vertex_x'], t.vertex_hit['x'])
        track_dict['vertex_y'] = np.append(track_dict['vertex_y'], t.vertex_hit['y'])
        track_dict['vertex_z'] = np.append(track_dict['vertex_z'], t.vertex_hit['z'])
        track_dict['end_x'] = np.append(track_dict['end_x'], t.end_hit['x'])
        track_dict['end_y'] = np.append(track_dict['end_y'], t.end_hit['y'])
        track_dict['end_z'] = np.append(track_dict['end_z'], t.end_hit['z'])

    tree.extend(track_dict)
    pass
