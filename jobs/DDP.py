import logging
import os
import pathlib
from datetime import datetime
import platform

import pandas as pd
import torch

import wandb

from utility.Control import load_config, cfg, save_config
from utility.Trainer import Trainer
from utility.EverythingNeeded import build_model, build_optimizer, build_loss, config_logging
from utility.FunctionTime import timing_decorator, print_accumulated_times

import torch.distributed as dist
import torch.multiprocessing as mp


from visualization.scripts.plotting import read_local_csv, visual_summary_link, visual_summary_momentum


def setup(rank, world_size):
    logger = logging.getLogger("GPU Setup")

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # check OS
    # Check if CUDA is available
    if torch.cuda.is_available():
        logger.info("CUDA is available.")

        # Check the number of available GPUs
        num_gpus = torch.cuda.device_count()
        logger.info(f"Number of available GPUs: {num_gpus}")

        # Print details for each GPU
        for i in range(num_gpus):
            gpu_info = torch.cuda.get_device_properties(i)
            logger.info(f"GPU {i}: {gpu_info.name}")

        # initialize the process group
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

        device = torch.device("cuda:0")
    # elif torch.has_mps:
    #     print("mps is available.")
    #     device = torch.device("mps")
    else:
        logger.info("CUDA is not available.")
        device = torch.device("cpu")

    cfg['device'] = device


def cleanup():
    dist.destroy_process_group()


@timing_decorator
def load_model_summary():
    logger = logging.getLogger("Main Process")

    model_dir = os.path.join(cfg['output_dir'], 'model.checkpoints')
    if not os.path.exists(model_dir):
        logger.info(f"==> No existing model output directory found, start from scratch.")
        return None, None
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth.tar')]
    model_exists = -1
    if model_files:
        epochs = sorted([int(file.split('.')[0].split('_')[-1]) for file in model_files], reverse=True)
        if epochs:
            model_exists = epochs[0]

    if model_exists >= 0:
        summary_path = os.path.join(cfg['output_dir'], 'summaries_0.csv')
        summary_log = pd.read_csv(summary_path) if os.path.exists(summary_path) else None
        if summary_log is not None:
            last_epoch = summary_log['epoch'].iloc[-1]
            model_exists = min(last_epoch, model_exists)
            summary_log = summary_log[summary_log['epoch'] <= model_exists]
            existed_model_path = os.path.join(model_dir, f'model_checkpoint_{model_exists:03d}.pth.tar')
            logger.info(f"==> Found existing model at epoch {model_exists}, loading it.")
            return existed_model_path, summary_log
        else:
            logger.info(f"==> No summary log found.")
    logger.info(f"==> No existing model found, start from scratch.")
    return None, None


@timing_decorator
def process(rank, world_size, config_path, verbose, record):
    load_config(config_path)
    config_logging(verbose, output_dir=cfg['output_dir'], rank=rank)

    logger = logging.getLogger("Main Process")

    logger.info(f"==> Running basic DDP on rank {rank} with total size {world_size}.")
    setup(rank, world_size)

    wandb.login(key="0d27159d2932514bfafad627aaee6c6a9a0ffc8d")

    c = datetime.now()
    current_time = c.strftime('%m/%d/%Y-%H:%M:%S')

    wandb.init(
        project=f"TrkGNN_DDP_{cfg['task']}",
        group=f"{pathlib.PurePath(cfg['output_dir']).name}@{current_time}",
        name=f"DDP_{rank}",
        job_type=f"{platform.node()}",
        # resume="auto",
        config=cfg,
        mode="online" if record else "disabled",
        dir=os.path.abspath(cfg['output_dir']),
        notes=cfg['notes'],
    )
    # define a metric we are interested in the minimum of
    wandb.define_metric("valid_loss", summary="min")
    # define artifact
    artifact = wandb.Artifact("best_model", type="model") if rank == 0 else None

    # check if the model and summary log exists, if so, load it
    existed_model_path, summary_log = load_model_summary()

    checkpoint_path = os.path.join(cfg['output_dir'], 'model.checkpoint')
    # Build model
    if torch.cuda.is_available():
        model = build_model(rank, distributed=True, existed_model_path=existed_model_path)
        # compile the model
        # model = torch_geometric.compile(model, dynamic=True)

        if rank == 0:
            # All processes should see same parameters as they all start from same
            # random parameters and gradients are synchronized in backward passes.
            # Therefore, saving it in one process is sufficient.
            torch.save(model.state_dict(), checkpoint_path)

        # Use a barrier() to make sure that process 1 loads the model after process
        # 0 saves it.
        dist.barrier()
        # configure map_location properly
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        model.load_state_dict(torch.load(checkpoint_path, map_location=map_location))

        # Check if the model is on the correct GPU
        def is_model_on_device(m, device):
            for param in m.parameters():
                if param.device != device:
                    return param.device
            return device

        if d := is_model_on_device(model, rank):
            logger.debug(f"Model is on the correct device: {d}")
        else:
            logger.debug(f"Model is not on the correct device. Expected device: {d}")

    else:
        model = build_model(cfg['device'], distributed=False, existed_model_path=existed_model_path)

    # Back to normal training
    optimizer, lr_scheduler = build_optimizer(
        model.parameters(),
        n_rank=world_size,
        lr_warmup_epochs=1,
        existed_optimizer_path=existed_model_path,
        last_epoch=summary_log['epoch'].iloc[-1] if summary_log is not None else -1,
        **cfg['optimizer']
    )
    loss = build_loss(cfg['loss_func'])

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        loss_func=loss,
        device=rank,
        distributed=torch.cuda.is_available(),
    )

    trainer.summaries = summary_log
    trainer.best_model_artifact = artifact

    # wandb.watch(
    #     trainer.model,
    #     criterion=trainer.loss_func_y if cfg['task'] == 'link' else trainer.loss_func_p,
    #     log='all',
    #     log_freq=1,
    # )

    trainer.process(
        n_epochs=cfg['training']['n_epochs'],
        n_total_epochs=cfg['training']['n_total_epochs'],
        world_size=world_size
    )

    if torch.cuda.is_available():
        if rank == 0:
            os.remove(checkpoint_path)

        cleanup()
        logger.info(f"==> Finish running basic DDP on rank {rank}.")
    else:
        logger.info(f"==> Finish running training on {cfg['device']}.")

    print_accumulated_times()
    # visualization the training summary
    if torch.cuda.is_available() and rank != 0:
        return
    os.makedirs(cfg['plot_path'], exist_ok=True)
    df, t = read_local_csv(os.path.join(cfg['output_dir'], 'summaries_0.csv'))

    fig = None
    if cfg['task'] == 'link':
        fig = visual_summary_link(df, t)
    elif cfg['task'] == 'momentum':
        fig = visual_summary_momentum(df, t)
    fig.write_image(os.path.join(cfg['plot_path'], 'training_summary.png'))
    fig.write_image(os.path.join(cfg['plot_path'], 'training_summary.pdf'))

    save_config(cfg)

    wandb.finish()


def parallel_process(config_path, world_size, verbose, record=False):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    if verbose:
        os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

    mp.spawn(
        process,
        args=(world_size, config_path, verbose, record),
        nprocs=world_size,
        join=True
    )


if __name__ == '__main__':
    load_config('/Users/avencast/PycharmProjects/trkgnn/configs/mpnn.yaml')

    config_logging(True, None)
    process('cpu', 1)

    print_accumulated_times()
