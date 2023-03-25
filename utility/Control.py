import os

import yaml

cfg = {
    "data": {
        "chunk_size": "500 MB",
        "E0": 1,
    }
}


def load_config(file_str: str) -> None:
    global cfg

    cfg['_config_abs_path'] = os.path.abspath(file_str)

    with open(file_str) as f:
        cfg.update(yaml.safe_load(f))


if __name__ == '__main__':
    load_config(r'scripts/config.yaml')

    print(yaml.dump(cfg, default_flow_style=False))
