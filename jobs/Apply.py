import gc
import logging
import os
from typing import Tuple

import pandas as pd
import torch
import torch_geometric

from utility.Control import cfg
from utility.DataLoader import get_data_loaders
from utility.EverythingNeeded import config_logging, build_model, build_loss
from utility.FunctionTime import timing_decorator


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

        itr = 0
        while True:
            try:
                apply_loader = next(data_generator)
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
                    df_edge, df_node = unbatch_graphs(batch, y_pred, p_pred)

                    df_edge_list.append(df_edge)
                    df_node_list.append(df_node)

                itr += 1
            except StopIteration:
                print("Finish")
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


@timing_decorator
def unbatch_graphs(batch, y_pred, p_pred=None):
    all_edge_dfs = []
    all_node_dfs = []
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

    return pd.concat(all_edge_dfs).reset_index(drop=True), pd.concat(all_node_dfs).reset_index()
