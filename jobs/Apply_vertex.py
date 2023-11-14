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

from utility.Control import cfg, load_config
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
def load_vertex(model_dir: str, config_dir: str):
    load_config(config_dir)

    setup()
    logger = logging.getLogger("Apply.Vertex")

    logger.info(f"vertex model dir: {model_dir}")

    # Build model and load model
    model = build_model(cfg['device'], distributed=False)
    tar = torch.load(model_dir, map_location=cfg['device'])
    model.load_state_dict(tar['model'])
    model.to(cfg['device'])
    model.eval()

    return model
