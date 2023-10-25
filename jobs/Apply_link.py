import gc
import logging
import warnings
import os
import torch
import torch_geometric

from utility.Control import cfg
from utility.DataLoader import get_data_loaders
from utility.EverythingNeeded import config_logging, build_model
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
    # cfg['data']['read_from_graph'] = False

    # Run all events
    if 'global_stop' in cfg['data']:
        del cfg['data']['global_stop']


@timing_decorator
@torch.no_grad()
def predict(input_dir: list[str], model_dir: str, output_dir: str):
    config_logging(True, output_dir=output_dir, prefix='apply')
    setup()
    logger = logging.getLogger("Apply.Link")

    logger.info(f"input_dir: {input_dir}")
    logger.info(f"model_dir: {model_dir}")
    logger.info(f"output_dir: {output_dir}")

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

        output_graph_dir = os.path.join(output_dir, f"predicted_graphs")
        os.makedirs(output_graph_dir, exist_ok=True)

        itr = 0
        while True:
            try:
                apply_loader = next(data_generator)
                logger.info(f"Processing {itr + 1}th iteration with {len(apply_loader)} batches.")

                predicted_graph_list = []
                # Loop over batches
                for j, batch in enumerate(apply_loader):
                    batch = batch.to(cfg['device'])
                    batch_out = model(batch)
                    y_pred = torch.sigmoid(batch_out)
                    batch.edge_attr = torch.cat([batch.edge_attr, y_pred.unsqueeze(-1)], dim=1)

                    predicted_graph_list += [batch.to_data_list()]

                torch.save(predicted_graph_list, os.path.join(output_graph_dir, f"graph_{itr}.pt"))
                itr += 1
            except StopIteration:
                print("Finish")
                # logout to root file
                logger.info(f"Finish predicting file: {data_dir}")
                break
