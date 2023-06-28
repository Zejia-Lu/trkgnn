"""
Module containing a pytorch graph network implementation modeled after
DeepMind's InteractionNetwork with Residual connections.
"""

# Externals
import torch
import torch.nn as nn
from torch_scatter import scatter_add
from torch.utils.checkpoint import checkpoint

# Locals
from .utils import make_mlp

from utility.EverythingNeeded import get_memory_size_MB, print_gpu_info


class GNN(nn.Module):
    """
    A message-passing graph network which takes a graph with:
    - bi-directional edges
    - node features, no edge features

    and applies the following modules:
    - a graph encoder (no message passing)
    - recurrent edge and node networks
    - an edge classifier
    """

    def __init__(self, input_dim, hidden_dim,
                 n_encoder_layers=2, n_edge_layers=4, n_node_layers=4,
                 n_graph_iters=1, layer_norm=True):
        super(GNN, self).__init__()
        self.n_graph_iters = n_graph_iters

        # The node encoder transforms input node features to the hidden space
        self.node_encoder = make_mlp(input_dim, [hidden_dim] * n_encoder_layers)

        # The edge network computes new edge features from connected nodes
        self.edge_network = make_mlp(2 * hidden_dim, [hidden_dim] * n_edge_layers,
                                     layer_norm=layer_norm)

        # The node network computes new node features
        self.node_network = make_mlp(2 * hidden_dim, [hidden_dim] * n_node_layers,
                                     layer_norm=layer_norm)

        # The edge classifier computes final edge scores
        self.edge_classifier = make_mlp(
            2 * hidden_dim, [hidden_dim, 1], output_activation=None
        )

        # The momentum change regressor computes the momentum change for each edge
        self.momentum_change_regressor = make_mlp(
            2 * hidden_dim, [hidden_dim, 1], output_activation=None
        )

        # solve memory issue
        self._set_static_graph(True)

    def forward(self, data, verbose=False):
        # Make every edge bi-directional
        send_idx = torch.cat([data.edge_index[0], data.edge_index[1]], dim=0)
        recv_idx = torch.cat([data.edge_index[1], data.edge_index[0]], dim=0)

        if verbose:
            print_gpu_info(prefix="GNN forward")

        # Encode the graph features into the hidden space
        x = self.node_encoder(data.x)

        if verbose:
            print_gpu_info(prefix="node_encoder")

        # We can move the checkpointing to this level
        def computation(x):
            for i in range(self.n_graph_iters):
                # Previous hidden state
                x0 = x

                if verbose:
                    print_gpu_info(prefix="graph iteration {}: 1".format(i))
                # Compute new edge features
                edge_inputs = torch.cat([x[send_idx], x[recv_idx]], dim=1)
                if verbose:
                    print_gpu_info(prefix="graph iteration {}: 2".format(i))
                e = self.edge_network(edge_inputs)
                if verbose:
                    print_gpu_info(prefix="graph iteration {}: 3".format(i))

                # Sum edge features coming into each node
                aggr_messages = scatter_add(e, recv_idx, dim=0, dim_size=x.shape[0])
                if verbose:
                    print_gpu_info(prefix="graph iteration {}: 4".format(i))
                # Compute new node features
                node_inputs = torch.cat([x, aggr_messages], dim=1)
                if verbose:
                    print_gpu_info(prefix="graph iteration {}: 5".format(i))
                x = self.node_network(node_inputs)
                if verbose:
                    print_gpu_info(prefix="graph iteration {}: 6".format(i))
                # Residual connection
                x = x + x0
                if verbose:
                    print_gpu_info(prefix="graph iteration {}: 7".format(i))

            return x

        # Call computation with checkpoint
        x = checkpoint(computation, x)

        # Compute final edge scores; use original edge directions only
        start_idx, end_idx = data.edge_index
        clf_inputs = torch.cat([x[start_idx], x[end_idx]], dim=1)
        edge_scores = self.edge_classifier(clf_inputs).squeeze(-1)

        # Compute momentum change for each edge
        momentum_changes = self.momentum_change_regressor(clf_inputs).squeeze(-1)

        return edge_scores, momentum_changes


def build_model(**kwargs):
    return GNN(**kwargs)
