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
        set_default(cfg, 'notes', "Empty comment.")
        set_default(cfg, 'num_track_predict', True)
        set_default(cfg, 'edge_features', False)
        set_default(cfg, 'momentum_predict', False)
        set_default(cfg['data'], 'E0', 1)
        set_default(cfg['data'], 'chunk_size', "10 MB")
        set_default(cfg['data'], 'read_from_graph', False)
        set_default(cfg['data'], 'global_stop_graph_file', -1)
        set_default(cfg['data'], 'graph_with_BField', True)
        set_default(cfg['data'], 'scale_factor_BField', 100.0)
        set_default(cfg['data'], 'min_graph_size', 1)
        set_default(cfg['data'], 'max_graph_size', 80)

        set_default(cfg['training'], 'n_epochs', cfg['training']['n_total_epochs'])

        cfg['plot_path'] = os.path.join(cfg['output_dir'], 'plots')


def save_config(config) -> None:
    cfg_path = os.path.join(config['output_dir'], 'train.yaml')

    if not os.path.exists(config['output_dir']):
        return

    with open(cfg_path, 'w') as f:
        yaml.dump(config, f)


if __name__ == '__main__':
    load_config(r'scripts/config.yaml')

    print(yaml.dump(cfg, default_flow_style=False))
