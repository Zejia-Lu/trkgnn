import os

from utility.Control import load_config, cfg
from utility.Trainer import Trainer
from utility.EverythingNeeded import build_model, build_optimizer, build_loss, config_logging
from utility.FunctionTime import timing_decorator, print_accumulated_times

import torch.distributed as dist
import torch.multiprocessing as mp


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


@timing_decorator
def process(rank, world_size, config_path):
    load_config(config_path)
    config_logging(True, output_dir=cfg['output_dir'], rank=rank)

    print(f"==> Running basic DDP on rank {rank} with total size {world_size}.")
    setup(rank, world_size)

    model = build_model(rank, world_size > 1)
    optimizer, lr_scheduler = build_optimizer(model.parameters(), **cfg['optimizer'])
    loss = build_loss(cfg['loss_func'])

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        loss_func=loss,
        device=rank,
        distributed=False
    )

    trainer.process(
        n_epochs=cfg['training']['n_total_epochs'],
        n_total_epochs=cfg['training']['n_total_epochs'],
        rank=rank, world_size=world_size
    )

    cleanup()
    print(f"==> Finish running basic DDP on rank {rank}.")
    print_accumulated_times()


def parallel_process(config_path, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    mp.spawn(process,
             args=(world_size, config_path),
             nprocs=world_size,
             join=True)


if __name__ == '__main__':
    load_config('/Users/avencast/PycharmProjects/trkgnn/configs/mpnn.yaml')

    config_logging(True, None)
    process('cpu', 1)

    print_accumulated_times()
