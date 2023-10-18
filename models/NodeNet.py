# Externals
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

# Locals
from .utils import make_mlp


class NodeNetwork(MessagePassing):
    def __init__(self, in_channels, out_channels, n_node_layers, layer_norm=False):
        super(NodeNetwork, self).__init__(aggr='max')  # "Add" aggregation
        self.node_network = make_mlp(2 * in_channels, [out_channels] * n_node_layers, layer_norm=layer_norm)

    def forward(self, x, e, edge_index):
        # x: Node features, [num_nodes, num_features]
        # edge_index: Edge indices, [2, num_edges]

        return self.propagate(edge_index, x=x, e=e)

    def message(self, x_i, x_j, e):
        # x_i: Source node features, [num_edges, num_features]
        # x_j: Target node features, [num_edges, num_features]

        # Concatenate source and target node features
        # edge_features = torch.cat([x_i, x_j - x_i, e_j], dim=-1)
        edge_features = torch.cat([x_i, e - x_i], dim=-1)

        # Compute new edge features using the node network
        return self.node_network(edge_features)
