import argparse
from utility.Control import load_config

from jobs.quick_test import quick_test
from jobs.DDP import parallel_process
from jobs.Apply import apply_to_ds
from jobs.dummy_test import dummy_test
from utility.FunctionTime import print_accumulated_times


def main(arg):
    if arg.command == 'test':
        load_config(arg.config)
        quick_test()

    if arg.command == 'DDP':
        parallel_process(arg.config, args.world_size, args.verbose)

    if arg.command == 'apply':
        load_config(arg.config)
        apply_to_ds(arg.input, arg.model, arg.output, arg.save)
        print_accumulated_times()

    if arg.command == 'dummy':
        load_config(arg.config)
        dummy_test()


if __name__ == '__main__':
    from pathlib import Path
    import sys

    base_dir = Path(__file__).parent.resolve().parent.absolute()
    sys.path.append(f'{base_dir}')

    par = argparse.ArgumentParser(prog='GNN', description='GNN Tracking Toolkit')
    subparsers = par.add_subparsers(title='modules', help='sub-command help', dest='command')

    # parser for quick-test
    test = subparsers.add_parser('test', help='quick test using small dataset')
    test.add_argument('config', default='config.yaml', type=str, help="the config file")

    # parser for quick-test
    DDP = subparsers.add_parser('DDP', help='Distributed Data Parallel Training')
    DDP.add_argument('config', default='config.yaml', type=str, help="the config file")
    DDP.add_argument('-w', '--world_size', type=int, default=1)
    DDP.add_argument('-v', '--verbose', action='store_true')

    # parser for evaluation/application
    apply = subparsers.add_parser('apply', help='apply the model to the dataset')
    # add argument for the input dataset (can be multiple)
    apply.add_argument('input', nargs='+', type=str, help="the input dataset directories")
    # add argument for the model directory
    apply.add_argument('-m', '--model', default='model', type=str, help="the model directory")
    # add argument for the output directory (default: current directory)
    apply.add_argument('-o', '--output', default='.', type=str, help="the output directory")
    # add argument for training config file
    apply.add_argument('-c', '--config', default='config.yaml', type=str, help="the config file for training")
    # add argument for saving graphs
    apply.add_argument('-s', '--save', action='store_true', help="save the graphs to the output directory")

    # parser for dummy test
    dummy = subparsers.add_parser('dummy', help='dummy test the evaluation speed')
    dummy.add_argument('config', default='config.yaml', type=str, help="the config file")

    args = par.parse_args()

    main(args)
