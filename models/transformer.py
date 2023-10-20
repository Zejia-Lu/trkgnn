# Externals
import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv
from torch_scatter import scatter_add

# Locals
from .utils import make_mlp


class TransGNN(nn.Module):
    def __init__(self, node_input_dim, edge_input_dim, hidden_dim, heads=4, n_encoder_layers=2, n_iterations=1):
        super(TransGNN, self).__init__()

        self.n_iterations = n_iterations

        self.node_embedding = make_mlp(node_input_dim, [hidden_dim] * n_encoder_layers)
        self.edge_embedding = make_mlp(edge_input_dim, [hidden_dim] * n_encoder_layers)

        self.transformer_conv_list = nn.ModuleList([
            TransformerConv(2 * hidden_dim, hidden_dim, heads=heads) for _ in range(n_iterations)
        ])

        self.norm_transconv_list = nn.ModuleList([
            nn.LayerNorm(heads * hidden_dim) for _ in range(n_iterations)
        ])

        self.norm_combined_list = nn.ModuleList([
            nn.LayerNorm(2 * hidden_dim) for _ in range(n_iterations)
        ])

        self.projection_layer_edge_list = nn.ModuleList(
            [nn.Linear(hidden_dim * heads, hidden_dim) for _ in range(n_iterations)]
        )
        self.projection_layer_node_list = nn.ModuleList(
            [nn.Linear(hidden_dim * heads, hidden_dim) for _ in range(n_iterations)]
        )

        # self.transformer_conv = TransformerConv(2 * hidden_dim, hidden_dim, heads=heads)
        # self.norm_transconv = nn.LayerNorm(heads * hidden_dim)
        # self.norm_combined = nn.LayerNorm(2 * hidden_dim)
        # self.projection_layer_edge = nn.Linear(hidden_dim * heads, hidden_dim)
        # self.projection_layer_node = nn.Linear(hidden_dim * heads, hidden_dim)

        # The edge classifier computes final edge scores
        self.edge_classifier = make_mlp(2 * hidden_dim, [hidden_dim, 1], output_activation=None)

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
                transformer_conv, norm_combined, norm_transconv, projection_layer_edge,
                projection_layer_node) in enumerate(
            zip(
                self.transformer_conv_list,
                self.norm_combined_list,
                self.norm_transconv_list,
                self.projection_layer_edge_list,
                self.projection_layer_node_list
            )):
            x0 = node_features
            e0 = edge_features

            # Aggregate edge features to nodes
            aggregated_from_src = scatter_add(edge_features, dst_indices, dim=0, dim_size=node_features.shape[0])

            # Combine node and aggregated edge features
            combined_features = torch.cat([node_features, aggregated_from_src - node_features], dim=1)
            combined_features = norm_combined(combined_features)
            # print(f"Norm Combined Layer {i} is called. Output has grad_fn: {combined_features.requires_grad}")

            # Pass through Transformer layer
            out_node_features = transformer_conv(combined_features, edge_indices)
            # print(f"Transformer Conv Layer {i} is called. Output has grad_fn: {out_node_features.requires_grad}")

            out_node_features = norm_transconv(out_node_features)
            # print(f"Norm TransConv Layer {i} is called. Output has grad_fn: {out_node_features.requires_grad}")

            # Update node and edge features for the next iteration
            node_features = projection_layer_node(out_node_features)
            # print(f"Projection Layer Node {i} is called. Output has grad_fn: {node_features.requires_grad}")

            edge_features = projection_layer_edge(out_node_features[src_indices] - out_node_features[dst_indices])
            # print(f"Projection Layer Edge {i} is called. Output has grad_fn: {edge_features.requires_grad}")

            # shortcut
            node_features = node_features + x0
            edge_features = edge_features + e0

        # Compute final edge scores; use original edge directions only
        start_idx, end_idx = data.edge_index
        clf_inputs = torch.cat([node_features[start_idx], node_features[end_idx]], dim=1)
        edge_output = self.edge_classifier(clf_inputs).squeeze(-1)

        return edge_output


def build_model(**kwargs):
    return TransGNN(**kwargs)
