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
def predict(input_dir: list[str], model_dir: str, output_dir: str, truth: bool = False, vtx_model=None):
    config_logging(True, output_dir=output_dir, prefix='apply')
    setup()
    logger = logging.getLogger("Apply.Link")

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
                logger.info(f"Processing chunk {itr}")
                next_chunk = next(data_generator)
                if isinstance(next_chunk, tuple):
                    apply_loader, chunk_name = next_chunk
                else:
                    apply_loader, chunk_name = next_chunk, None
                if chunk_name:
                    logger.info(f"Chunk source file: {chunk_name}")
                logger.info(f"Chunk {itr} has {len(apply_loader)} batches")
                predicted_graph_list = []

                for j, batch in enumerate(apply_loader):
                    batch = batch.to(cfg['device'])
                    batch_out = model(batch)
                    # y_pred = torch.sigmoid(batch_out)
                    
                    # Version final
                    y_pred, edge_feats = batch_out
                    y_pred = torch.sigmoid(y_pred)

                    if not truth:
                        batch.edge_attr = torch.cat([batch.edge_attr, y_pred.unsqueeze(-1)], dim=1)
                    else:
                        batch.edge_attr = torch.cat([batch.edge_attr, batch.y.unsqueeze(-1)], dim=1)

                    batch = batch.to("cpu")
                    graphs = batch.to_data_list()
                    predicted_graph_list += graphs

                    del batch, batch_out, y_pred
                    torch.cuda.empty_cache()
                    import gc
                    gc.collect()

                output_name = chunk_name if chunk_name else f"graph_{itr}.pt"
                torch.save(predicted_graph_list, os.path.join(output_graph_dir, output_name))
                logger.info(f"Saved output file: {output_name}")
                itr += 1

            except StopIteration:
                logger.info(f"Finished processing {data_dir} with {itr} chunks")
                break

            except Exception as e:
                logger.error(f"Error processing chunk {itr} (attempting again): {e}")
                continue
                
            
