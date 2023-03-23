import argparse
from utility.Control import load_config

from jobs.quick_test import quick_test


def main(arg):
    if arg.command == 'test':
        load_config(arg.config)
        quick_test()

    pass


if __name__ == '__main__':
    from pathlib import Path
    import sys

    base_dir = Path(__file__).parent.resolve().parent.absolute()
    sys.path.append(f'{base_dir}')

    par = argparse.ArgumentParser(prog='GNN', description='GNN Tracking Toolkit')
    subparsers = par.add_subparsers(title='modules', help='sub-command help', dest='command')

    # parser for init
    test = subparsers.add_parser('test', help='quick test using small dataset')
    test.add_argument('config', default='config.yaml', type=str, help="the config file")

    args = par.parse_args()

    main(args)
