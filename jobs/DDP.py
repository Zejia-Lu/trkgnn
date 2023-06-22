import os

import torch

from utility.Control import load_config, cfg
from utility.Trainer import Trainer
from utility.EverythingNeeded import build_model, build_optimizer, build_loss, config_logging
from utility.FunctionTime import timing_decorator, print_accumulated_times

import torch.distributed as dist
import torch.multiprocessing as mp

from visualization.scripts.plotting import read_local_csv, visual_summary_log


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # check OS
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

        # initialize the process group
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

        device = torch.device("cuda:0")
    elif torch.has_mps:
        print("mps is available.")
        device = torch.device("mps")
    else:
        print("CUDA is not available.")
        device = torch.device("cpu")

    cfg['device'] = device


def cleanup():
    dist.destroy_process_group()


@timing_decorator
def process(rank, world_size, config_path, verbose):
    load_config(config_path)
    config_logging(verbose, output_dir=cfg['output_dir'], rank=rank)

    print(f"==> Running basic DDP on rank {rank} with total size {world_size}.")
    setup(rank, world_size)

    checkpoint_path = os.path.join(cfg['output_dir'], 'model.checkpoint')
    # Build model
    if torch.cuda.is_available():
        model = build_model(rank, distributed=True)
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
            print(f"Model is on the correct device: {d}")
        else:
            print(f"Model is not on the correct device. Expected device: {d}")

    else:
        model = build_model(cfg['device'], distributed=False)

    # Back to normal training
    optimizer, lr_scheduler = build_optimizer(model.parameters(), n_rank=world_size, **cfg['optimizer'])
    loss = build_loss(cfg['loss_func'])

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        loss_func=loss,
        device=rank,
        distributed=torch.cuda.is_available(),
    )

    trainer.process(
        n_epochs=cfg['training']['n_total_epochs'],
        n_total_epochs=cfg['training']['n_total_epochs'],
        world_size=world_size
    )

    if torch.cuda.is_available():
        if rank == 0:
            os.remove(checkpoint_path)

        cleanup()
        print(f"==> Finish running basic DDP on rank {rank}.")
    else:
        print(f"==> Finish running training on {cfg['device']}.")

    # visualization the training summary
    os.makedirs(cfg['plot_path'], exist_ok=True)
    df, t = read_local_csv(os.path.join(cfg['output_dir'], 'summaries_0.csv'))
    fig = visual_summary_log(df, t)
    fig.write_image(os.path.join(cfg['plot_path'], 'training_summary.png'))
    fig.write_image(os.path.join(cfg['plot_path'], 'training_summary.pdf'))

    print_accumulated_times()


def parallel_process(config_path, world_size, verbose):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    mp.spawn(process,
             args=(world_size, config_path, verbose),
             nprocs=world_size,
             join=True)


if __name__ == '__main__':
    load_config('/Users/avencast/PycharmProjects/trkgnn/configs/mpnn.yaml')

    config_logging(True, None)
    process('cpu', 1)

    print_accumulated_times()
