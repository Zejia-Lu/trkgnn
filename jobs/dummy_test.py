import time

import torch_geometric

from utility.EverythingNeeded import config_logging, build_model
from utility.Control import cfg

import torch
from torch_geometric.utils import erdos_renyi_graph
from torch_geometric.loader import DataLoader


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
    if 'global_stop' in cfg['data']:
        del cfg['data']['global_stop']


def create_dummy_graphs(num_graphs, start_size, end_size, num_node_features):
    data_list = []
    for i in range(num_graphs):
        n = torch.randint(start_size, end_size, (1,)).item()
        edge_index = erdos_renyi_graph(n, edge_prob=0.5, directed=False)
        x = torch.randn((n, num_node_features))  # random node features
        data = torch_geometric.data.Data(x=x, edge_index=edge_index)
        data_list.append(data)
    return data_list


def evaluate_speed(num_graphs, start_size, end_size, model, batch_size=16):
    # Generate dummy data
    data_list = create_dummy_graphs(num_graphs, start_size, end_size, 4)
    loader = DataLoader(data_list, batch_size=batch_size, shuffle=True)

    times = []
    for graph in loader:
        graph = graph.to(cfg['device'])
        start_time = time.time()
        _ = model(graph)
        end_time = time.time()
        times.append(end_time - start_time)

    mean_speed = sum(times) / len(data_list)
    print(f"Mean speed: {mean_speed} s [{len(data_list)} graphs]")


def dummy_test():
    setup()

    # Build model and load model
    model = build_model(cfg['device'], distributed=False)
    model.to(cfg['device'])
    model.eval()

    evaluate_speed(5000, 6, 20, model, 16)

    evaluate_speed(5000, 60, 500, model, 2)

    pass
