"""Utility functions for building models"""

import torch.nn as nn


def make_mlp(
        input_size, sizes,
        hidden_activation='ReLU',
        output_activation='ReLU',
        layer_norm=False,
        dropout=0
):
    """Construct an MLP with specified fully-connected layers."""

    hidden_activation = getattr(nn, hidden_activation)
    if output_activation is not None:
        output_activation = getattr(nn, output_activation)
    layers = []
    n_layers = len(sizes)
    sizes = [input_size] + sizes

    # Hidden layers
    for i in range(n_layers - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if layer_norm:
            layers.append(nn.LayerNorm(sizes[i + 1]))
        layers.append(hidden_activation())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

    # Final layer
    layers.append(nn.Linear(sizes[-2], sizes[-1]))
    if output_activation is not None:
        if layer_norm:
            layers.append(nn.LayerNorm(sizes[-1]))
        layers.append(output_activation())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)

