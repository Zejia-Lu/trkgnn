"""
Module containing a pytorch graph network implementation modeled after
DeepMind's InteractionNetwork with Residual connections.
"""

# Externals
import torch
import torch.nn as nn

from .NodeNet import NodeNetwork
# Locals
from .utils import make_mlp


class GNN(nn.Module):
    """
    A message-passing graph network which takes a graph with:
    - bidirectional edges
    - node features, no edge features

    and applies the following modules:
    - a graph encoder (no message passing)
    - recurrent edge and node networks
    - an edge classifier
    """

    def __init__(self, input_dim, hidden_dim,
                 n_encoder_layers=2, n_edge_layers=4, n_node_layers=4,
                 n_graph_iters=1, layer_norm=True, snf_output_dim=None):
        super(GNN, self).__init__()
        self.n_graph_iters = n_graph_iters

        # The node encoder transforms input node features to the hidden space
        self.node_encoder = make_mlp(input_dim, [hidden_dim] * n_encoder_layers)

        # The edge network computes new edge features from connected nodes
        self.edge_network = make_mlp(2 * hidden_dim, [hidden_dim] * n_edge_layers, layer_norm=layer_norm)

        # The node network computes new node features
        # self.node_network = make_mlp(2 * hidden_dim, [hidden_dim] * n_node_layers, layer_norm=layer_norm)
        self.node_network = NodeNetwork(hidden_dim, hidden_dim, n_node_layers, layer_norm=layer_norm)

        # The edge classifier computes final edge scores
        self.edge_classifier = make_mlp(2 * hidden_dim, [hidden_dim, 1], output_activation=None)

        # Add this line to accept the additional argument for SNF output dimension
        self.snf_output_dim = snf_output_dim

        # Add an MLP to output features for the SNF model
        if snf_output_dim is not None:
            self.edge_snf_output = make_mlp(
                2 * hidden_dim, [hidden_dim, snf_output_dim], output_activation=None
            )

    def forward(self, data, verbose=False):
        # Make every edge bidirectional
        send_idx = torch.cat([data.edge_index[0], data.edge_index[1]], dim=0)
        recv_idx = torch.cat([data.edge_index[1], data.edge_index[0]], dim=0)

        # Encode the graph features into the hidden space
        x = self.node_encoder(data.x)

        # Loop over graph iterations
        for i in range(self.n_graph_iters):
            # Previous hidden state
            x0 = x

            # Compute new edge features
            edge_inputs = torch.cat([x[send_idx], x[recv_idx]], dim=1)
            e = self.edge_network(edge_inputs)

            # # Sum edge features coming into each node
            # aggr_messages = scatter_add(e, recv_idx, dim=0, dim_size=x.shape[0])
            #
            # # Compute new node features
            # node_inputs = torch.cat([x, aggr_messages], dim=1)
            # x = self.node_network(node_inputs)

            # Sum edge features coming into each node
            x = self.node_network(x, e[recv_idx], torch.cat((send_idx.view(1, -1), recv_idx.view(1, -1)), dim=0))

            # Residual connection
            x = x + x0

        # Compute final edge scores; use original edge directions only
        start_idx, end_idx = data.edge_index
        clf_inputs = torch.cat([x[start_idx], x[end_idx]], dim=1)
        edge_scores = self.edge_classifier(clf_inputs).squeeze(-1)

        # Compute features for the SNF model
        if self.snf_output_dim is not None:
            snf_output = self.edge_snf_output(clf_inputs)
            return edge_scores, snf_output
        else:
            return edge_scores


def build_model(**kwargs):
    return GNN(**kwargs)
