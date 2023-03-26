import os

import yaml

cfg = {}


def set_default(config, key, default_value):
    if key not in config:
        config[key] = default_value


def load_config(file_str: str) -> None:
    global cfg

    cfg['_config_abs_path'] = os.path.abspath(file_str)

    with open(file_str) as f:
        cfg.update(yaml.safe_load(f))

        # default value setting
        set_default(cfg, 'momentum_predict', False)
        set_default(cfg['data'], 'E0', 1)
        set_default(cfg['data'], 'chunk_size', "10 MB")


if __name__ == '__main__':
    load_config(r'scripts/config.yaml')

    print(yaml.dump(cfg, default_flow_style=False))
