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
        parallel_process(arg.config, args.world_size, args.verbose, record=args.record)

    if arg.command == 'apply':

        predict = None
        vtx_model = None
        if arg.process == 'link':
            from jobs.Apply_link import predict
        elif arg.process == 'momentum':
            from jobs.Apply_momentum import predict
        elif args.process == 'vertex':
            from jobs.Apply_vertex import load_vertex
            vtx_model = load_vertex(args.vertex_model, args.vertex_config)
            from jobs.Apply_momentum import predict
        load_config(args.config)
        predict(args.input, args.model, args.output, args.truth, vtx_model)
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
    DDP.add_argument('-r', '--record', action='store_true', help="record details using wandb")

    # parser for evaluation/application
    apply_new = subparsers.add_parser('apply', help='apply the link model to the dataset')
    # add argument for the input dataset (can be multiple)
    apply_new.add_argument('input', nargs='+', type=str, help="the input dataset directories")
    # add argument for the model directory
    apply_new.add_argument('-m', '--model', default='model', type=str, help="the model directory")
    # add argument for the output directory (default: current directory)
    apply_new.add_argument('-o', '--output', default='.', type=str, help="the output directory")
    # add argument for training config file
    apply_new.add_argument('-c', '--config', default='config.yaml', type=str, help="the config file for training")
    # add argument for saving graphs
    apply_new.add_argument('-t', '--truth', action='store_true', help="use truth edge information")
    apply_new.add_argument(
        '-p', '--process', required=True, choices=['link', 'momentum', 'vertex'],
        help="the process type: link or momentum"
    )
    apply_new.add_argument(
        '-v', '--vertex_model', default='model', type=str, help="the vertex model directory (optional)"
    )
    apply_new.add_argument(
        '-x', '--vertex_config', default='model', type=str, help="the vertex model config (optional)"
    )

    # parser for dummy test
    dummy = subparsers.add_parser('dummy', help='dummy test the evaluation speed')
    dummy.add_argument('config', default='config.yaml', type=str, help="the config file")

    args = par.parse_args()

    main(args)
