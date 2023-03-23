import torch
import torch_geometric
import uproot as up
import awkward as ak
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

# Locals
from models import get_model
from utility.Control import cfg
from utility.ModelSummary import summary

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
    elif torch.has_mps:
        print("mps is available.")
        device = torch.device("mps")
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
    collection = 'TagTrk1'
    test_n = 10
    with up.open(cfg['data']['input_dir']) as f:
        node = f[tree_name].arrays([f'{collection}_{i}' for i in ["x", "y", "z"]])
        edge = f[tree_name].arrays([f'{collection}_{i}' for i in ["start", "end"]])
        truth = f[tree_name].arrays([f'{collection}_{i}' for i in ["truth"]])

        data = []
        for i in range(test_n):
            graph_data = {
                'node': node[i],
                'edge': edge[i],
                'truth': truth[i],
            }

            out_df = {}
            for key, value in graph_data.items():
                df = ak.to_dataframe(value)
                df.columns = df.columns.str.replace(f"{collection}_", "")
                out_df[key] = df

            data.append(torch_geometric.data.Data(
                x=torch.from_numpy(out_df['node'].to_numpy(dtype='float32')),
                edge_index=torch.from_numpy(out_df['edge'].to_numpy(dtype='int64').transpose()),
                y=torch.from_numpy(out_df['truth'].to_numpy(dtype='float32').squeeze()),
            ))

        # start testing model
        loss_func = getattr(nn.functional, 'binary_cross_entropy_with_logits')
        model.eval()
        batch = data[0]
        batch = batch.to(device)
        model.zero_grad()
        batch_output = model(batch)
        batch_loss = loss_func(batch_output, batch.y)

        print("==> Loss: ", batch_loss)
        print("==> Input node size: ", data[0].x.shape)
        print("==> Input edge size: ", data[0].edge_index.shape)
        print("==> Input truth size: ", data[0].y.shape)
        print("==> Output size: ", batch_output.shape)


pass


def quick_test():
    print("==> Quick test")

    print("1. ==> Check Environment")
    device = check_env()

    print("2. ==> Check Model")
    model = check_model(device)

    print("3. ==> Check DataSet")
    check_data(model, device)

    pass
