import pandas as pd
import torch
import torch_geometric
import uproot as up
import awkward as ak
import seaborn as sns
from matplotlib import pyplot
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

# Locals
from models import get_model
from utility.Control import cfg


def check_env():
    # check pytorch vision
    print("pyTorch vision: ", torch.__version__)
    # Check if CUDA is available
    if torch.cuda.is_available():
        print("CUDA is available.")

        # Check the number of available GPUs
        num_gpus = torch.cuda.device_count()
        print(f"Number of available GPUs: {num_gpus}")

        # Print details for each GPU
        for i in range(num_gpus):
            gpu_info = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {gpu_info.name}")

        device = torch.device("cuda:0")
    # elif torch.has_mps:
    #     print("mps is available.")
    #     device = torch.device("mps")
    else:
        print("CUDA is not available.")
        device = torch.device("cpu")

    return device


def check_model(device):
    # load model arguments

    if 'model' in cfg:
        model_configs = cfg['model']
        name = model_configs.pop('name')
        model = get_model(name=name, **model_configs).to(device)

        print(model)
        print('Parameters: %i' % sum(p.numel() for p in model.parameters()))

        return model
    else:
        print("model is not missing in config.")
        return None

    pass


def check_data(model, device):
    tree_name = 'dp'
    collection = 'DigitizedTagTrk'
    test_n = 10
    with up.open('/Users/avencast/CLionProjects/darkshine-simulation/workspace/Tracker_GNN.root') as f:
        node = f[tree_name].arrays([f'{collection}_{i}' for i in ["x", "y", "z"]])
        edge = f[tree_name].arrays([f'{collection}_{i}' for i in ["start", "end"]])
        truth = f[tree_name].arrays([f'{collection}_{i}' for i in ["truth"]])
        weight = f[tree_name].arrays([f'{collection}_{i}' for i in ["weight"]])
        p = f[tree_name].arrays([f'{collection}_{i}' for i in ["p"]])

        data = []
        for i in range(test_n):
            graph_data = {
                'node': node[i],
                'edge': edge[i],
                'truth': truth[i],
                'weight': weight[i],
                'p': p[i]
            }

            out_df = {}
            for key, value in graph_data.items():
                df = ak.to_dataframe(value)
                df.columns = df.columns.str.replace(f"{collection}_", "")
                out_df[key] = df

            y = torch.from_numpy(out_df['truth'].to_numpy(dtype='float32').squeeze())
            truth_w = torch.from_numpy(out_df['weight'].to_numpy(dtype='float32').squeeze())
            w = y * (1 - truth_w) / truth_w + (1 - y) * (1 - truth_w)
            data.append(torch_geometric.data.Data(
                x=torch.from_numpy(out_df['node'].to_numpy(dtype='float32')),
                edge_index=torch.from_numpy(out_df['edge'].to_numpy(dtype='int64').transpose()),
                y=y,
                w=w,
                p=torch.from_numpy(out_df['p'].to_numpy(dtype='float32').squeeze()) / cfg['data']['E0'],
            ))

        # start testing model
        loss_func = getattr(nn.functional, 'binary_cross_entropy_with_logits')
        model.eval()
        batch = Batch.from_data_list(data)
        batch = batch.to(device)
        model.zero_grad()
        batch_output = model(batch)

        # calculate momentum prediction
        con_mask = (batch.y == 1)
        p_truth = batch.p[con_mask]
        p_out = batch_output[1][con_mask]
        p_pred = model.sample(p_out).squeeze()

        batch_loss = model.loss(loss_func, batch_output[0], batch.y, p_pred, p_truth, weight=batch.w)

        print('data: ', data)
        print('batch: ', batch)
        print('output', batch_output)

        def decode_edge_output(output, edge_index, batch_info):
            decoded_output = []
            source_batch_info = batch_info[edge_index[0]]
            target_batch_info = batch_info[edge_index[1]]
            edge_to_batch = source_batch_info * (source_batch_info == target_batch_info)
            unique_batches = edge_to_batch.unique()
            for batch_id in unique_batches:
                mask = edge_to_batch == batch_id
                graph_output = output[mask]
                decoded_output.append(graph_output)

            return decoded_output

        batch_output = decode_edge_output(batch_output[0], batch.edge_index, batch.batch)

        print("==> Loss: ", batch_loss.item() / batch.num_graphs)
        print("==> Input node size: ", data[0].x.shape)
        print("==> Input edge size: ", data[0].edge_index.shape)
        print("==> Input truth size: ", data[0].y.shape)
        print("==> Output size: ", len(batch_output[0]))

        def get_tensor_memory_size(tensor):
            return tensor.element_size() * tensor.nelement()

        def data_memory_size(d):
            total_memory_size = 0
            for key, value in d.__dict__.items():
                if isinstance(value, torch.Tensor):
                    total_memory_size += get_tensor_memory_size(value)
            return total_memory_size

        print('data[0]: ', data_memory_size(data[0]))
        print('batch: ', data_memory_size(batch))

        # for idx, b in enumerate(data):
        #     print(f"    {idx} ==> len: {len(b.y)}")
        # for idx, b in enumerate(batch_output):
        #     print(f"    {idx} ==> len: {len(b)}")

        # print(batch_output)
        # plot
        # df = convert_graph2df(data[0], batch_output)
        # fig, axs = pyplot.subplots(2, figsize=(12, 12))
        # plot_xyz(axs[0], df, "x", "z")
        # plot_xyz(axs[1], df, "y", "z")
        # fig.show()


def convert_graph2df(data, predict):
    return {
        'node': pd.DataFrame(data.x.cpu().detach(), columns=['x', 'y', 'z']),
        'edge': pd.DataFrame(torch.transpose(data.edge_index, 0, 1).cpu().detach(), columns=['start', 'end']),
        'truth': pd.DataFrame(data.y.cpu().detach(), columns=['truth']),
        'predict': pd.DataFrame(predict.cpu().detach(), columns=['predict']),
    }


def plot_xyz(ax, d, x, y, predict_cut=0.8):
    sns.set_style(style='whitegrid')
    sns.scatterplot(ax=ax, data=d['node'], x=x, y=y)

    for i in range(len(d['edge'])):
        if d['truth'].iloc[i]['truth'] != 0:
            # or d['predict'].iloc[i]['predict'] < predict_cut: continue
            sns.lineplot(
                ax=ax,
                x=[
                    d['node'].iloc[d['edge'].iloc[i][f'start']][x],
                    d['node'].iloc[d['edge'].iloc[i][f'end']][x]
                ],
                y=[
                    d['node'].iloc[d['edge'].iloc[i][f'start']][y],
                    d['node'].iloc[d['edge'].iloc[i][f'end']][y]
                ],
                color='black'
            )

        if d['predict'].iloc[i]['predict'] >= predict_cut:
            sns.lineplot(
                ax=ax,
                x=[
                    d['node'].iloc[d['edge'].iloc[i][f'start']][x],
                    d['node'].iloc[d['edge'].iloc[i][f'end']][x]
                ],
                y=[
                    d['node'].iloc[d['edge'].iloc[i][f'start']][y],
                    d['node'].iloc[d['edge'].iloc[i][f'end']][y]
                ],
                color='red',
            )


def quick_test():
    print("==> Quick test")

    print("1. ==> Check Environment")
    device = check_env()

    print("2. ==> Check Model")
    model = check_model(device)

    print("3. ==> Check DataSet")
    check_data(model, device)

    pass
