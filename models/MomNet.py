# Externals
import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv

# Locals
from .utils import make_mlp
from utility.FunctionTime import timing_decorator


class MomNet(nn.Module):
    def __init__(self, node_input_dim, edge_input_dim, hidden_dim, heads=4, n_encoder_layers=2, n_iterations=1):
        super(MomNet, self).__init__()

        self.n_iterations = n_iterations

        self.node_embedding = make_mlp(node_input_dim, [hidden_dim] * n_encoder_layers)
        self.edge_embedding = make_mlp(edge_input_dim, [hidden_dim] * n_encoder_layers)

        self.momentum_transformer_list = nn.ModuleList([
            TransformerConv(hidden_dim, int(hidden_dim / heads), heads=heads, edge_dim=hidden_dim) for _ in
            range(n_iterations)
        ])

        self.edge_mlp_list = nn.ModuleList([
            make_mlp(2 * hidden_dim, [hidden_dim] * n_encoder_layers) for _ in range(n_iterations)
        ])

        self.mom_layer_norm_list = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(n_iterations)
        ])

        self.edge_layer_norm_list = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(n_iterations)
        ])

        # The momentum change regressor computes the momentum change for each edge
        self.momentum_regressor = make_mlp(
            input_size=2 * hidden_dim,
            sizes=[hidden_dim, 1],
            output_activation=None
        )

        self.edge_transformer_list = TransformerConv(hidden_dim, int(hidden_dim / heads), heads=heads,
                                                     edge_dim=hidden_dim)
        # The edge classifier computes final edge scores
        self.edge_classifier = make_mlp(
            input_size=2 * hidden_dim,
            sizes=[hidden_dim, 1],
            output_activation=None
        )

    @timing_decorator
    def forward(self, data, verbose=False):
        # Make every edge bidirectional
        send_idx = torch.cat([data.edge_index[0], data.edge_index[1]], dim=0)
        recv_idx = torch.cat([data.edge_index[1], data.edge_index[0]], dim=0)

        # Concatenate original and reverse edge attributes
        edge_attr_bi = torch.cat([data.edge_attr, data.edge_attr], dim=0)

        edge_indices = torch.stack([send_idx, recv_idx], dim=0)

        # Embed node and edge features
        node_features = self.node_embedding(data.x)
        edge_features = self.edge_embedding(edge_attr_bi)

        # Edge indices is of shape [2, E], where E is the number of edges
        src_indices, dst_indices = edge_indices

        for i, (
                momentum_transformer,
                edge_mlp,
                mom_ln,
                edge_ln,
        ) in enumerate(zip(
            self.momentum_transformer_list,
            self.edge_mlp_list,
            self.mom_layer_norm_list,
            self.edge_layer_norm_list,
        )):
            x0 = node_features
            e0 = edge_features

            # Apply transformer conv --> momentum update
            combined_nodes = momentum_transformer(x=node_features, edge_index=edge_indices, edge_attr=edge_features)
            combined_nodes = mom_ln(combined_nodes)

            # Update edge features
            new_edge_features = torch.cat([combined_nodes[src_indices], combined_nodes[dst_indices]], dim=1)
            new_edge_features = edge_mlp(new_edge_features)

            # shortcut
            node_features = combined_nodes + x0
            edge_features = new_edge_features + e0

        start_idx, end_idx = data.edge_index
        # Momentum prediction
        momentum_change = self.momentum_regressor(
            torch.cat([node_features[start_idx], node_features[end_idx]], dim=1)
        ).squeeze(-1)


        final_node_features = self.edge_transformer_list(
            x=node_features, edge_index=edge_indices, edge_attr=edge_features
        )
        # Compute final edge scores; use original edge directions only
        edge_scores = self.edge_classifier(
            torch.cat([final_node_features[start_idx], final_node_features[end_idx]], dim=1)
        ).squeeze(-1)

        return momentum_change, edge_scores


def build_model(**kwargs):
    return MomNet(**kwargs)
