# Externals
import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv
from torch_scatter import scatter_add
from torch_sparse import SparseTensor

import logging

# Locals
from .utils import make_mlp
from utility.FunctionTime import timing_decorator


class GraphTransformerLayer(nn.Module):
    def __init__(self, node_dim, edge_dim, nhead):
        super(GraphTransformerLayer, self).__init__()

        # Node Attention
        self.node_trans = TransformerConv(in_channels=node_dim, out_channels=node_dim, heads=nhead, edge_dim=edge_dim)

        # Edge Feature Update
        self.edge_trans = make_mlp(edge_dim + 2 * node_dim * nhead, [edge_dim, edge_dim], layer_norm=True)

        # projection layer
        self.projection_node = nn.Linear(node_dim * nhead, node_dim)

    @timing_decorator
    def forward(self, node_features, edge_features, edge_indices):
        # Node-level attention
        node_features = self.node_trans(node_features, edge_indices, edge_features)

        # Edge-level attention
        # Gather corresponding node features for each edge
        source_node_features = node_features[edge_indices[0]]
        target_node_features = node_features[edge_indices[1]]

        # Hierarchical: conditioned on node features
        edge_features_conditioned = torch.cat((edge_features, source_node_features, target_node_features), dim=-1)
        edge_features = self.edge_trans(edge_features_conditioned)

        node_features = self.projection_node(node_features)

        return node_features, edge_features


class MomNet(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, nhead=4, num_layers=2, num_iterations=1, n_encoder_layers=2):
        super(MomNet, self).__init__()

        self.num_iterations = num_iterations

        self.node_embedding = make_mlp(node_dim, [hidden_dim] * n_encoder_layers)
        self.edge_embedding = make_mlp(edge_dim, [hidden_dim] * n_encoder_layers)

        self.layers = nn.ModuleList([
            GraphTransformerLayer(node_dim=hidden_dim, edge_dim=hidden_dim, nhead=nhead) for _ in range(num_layers)
        ])

        # Additional components can be added for `mom_net` and `agg_net`
        # For example, fully connected layers for momentum prediction
        self.fc_momentum = make_mlp(hidden_dim, [edge_dim, 1], layer_norm=True, output_activation=None)

        # self.agg_layer = make_mlp(node_dim + edge_dim + 1, [edge_dim, edge_dim], layer_norm=True)
        self.agg_layer = TransformerConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            heads=nhead,
            edge_dim=hidden_dim + 1
        )

        self.norm_combined = nn.LayerNorm(hidden_dim * nhead)
        # projection layer
        self.projection_node = nn.Linear(hidden_dim * nhead, hidden_dim)

        # The edge classifier computes final edge scores
        self.edge_classifier = make_mlp(2 * hidden_dim, [node_dim, 1], output_activation=None)

    @timing_decorator
    def forward(self, x, edge_attr, edge_index, edge_scores_logit):
        send_idx = torch.cat([edge_index[0], edge_index[1]], dim=0)
        recv_idx = torch.cat([edge_index[1], edge_index[0]], dim=0)
        edge_indices = torch.stack([send_idx, recv_idx], dim=0)
        half_idx = edge_index[0].shape[0]

        edge_features = torch.cat([edge_attr, edge_scores_logit], dim=-1)
        edge_features = torch.cat([edge_features, edge_features], dim=0)

        # Embed node and edge features
        node_features = self.node_embedding(x)
        edge_features = self.edge_embedding(edge_features)

        momenta = torch.zeros(len(edge_features), 1, device=edge_features.device)
        for i in range(self.num_iterations):
            for layer in self.layers:
                # print('before: ', node_features.shape, edge_features.shape)
                node_features, edge_features = layer(node_features, edge_features, edge_indices)
                # print('after: ', node_features.shape, edge_features.shape)

            # Predicting momenta for edge features
            momenta = self.fc_momentum(edge_features)

            # Calculate average in-place for the first half
            momenta[:half_idx].add_(momenta[half_idx:]).div_(2)
            # Create the list with two elements referring to the averaged tensor
            momenta = torch.cat([momenta[:half_idx], momenta[:half_idx]], dim=0)
            # `agg_net` would be another model or a set of layers
            new_edge_features = torch.cat((edge_features, momenta), dim=-1)
            node_features = self.agg_layer(node_features, edge_indices, new_edge_features)
            node_features = self.norm_combined(node_features)
            node_features = self.projection_node(node_features)

        start_idx, end_idx = edge_index
        clf_inputs = torch.cat([node_features[start_idx], node_features[end_idx]], dim=1)
        edge_output = self.edge_classifier(clf_inputs).squeeze(-1)

        return momenta[:half_idx].squeeze(
            -1), edge_output  # momentum, and the edge score as probability of being a true edge


class LinkNet(nn.Module):
    def __init__(self, node_input_dim, edge_input_dim, hidden_dim, heads=4, n_encoder_layers=2, n_iterations=1):
        super(LinkNet, self).__init__()

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

        # The edge classifier computes final edge scores
        self.edge_classifier = make_mlp(2 * hidden_dim, [hidden_dim, 1], output_activation=None)

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


class TrkTrans(nn.Module):
    def __init__(
            self, node_input_dim, edge_input_dim,
            hidden_dim, heads=4, mom_heads=2, n_encoder_layers=2,
            n_iterations=1, n_mom_iterations=1,
            mom_mask_threshold=0.75,
            link_only: bool = False
    ):
        super(TrkTrans, self).__init__()

        self.mom_mask_threshold = mom_mask_threshold
        self.link_only = link_only

        self.link = LinkNet(node_input_dim, edge_input_dim, hidden_dim, heads, n_encoder_layers, n_iterations)
        if not link_only:
            self.mom = MomNet(
                node_input_dim,
                edge_input_dim + 1,
                hidden_dim=hidden_dim,
                nhead=mom_heads,
                num_layers=n_mom_iterations
            )

        self.logger = logging.getLogger(__name__)

    @timing_decorator
    def forward(self, data, split: bool = False):
        # LinkNet to first probe the graph structure
        edge_scores = self.link(data)
        if self.link_only:
            return edge_scores

        edge_scores_logit = torch.sigmoid(edge_scores)

        # build adj matrix
        # n_nodes = data.num_nodes
        # adj_matrix = SparseTensor(
        #     row=data.edge_index[0],
        #     col=data.edge_index[1],
        #     value=edge_scores_logit,
        #     sparse_sizes=(n_nodes, n_nodes)
        # ).to_torch_sparse_coo_tensor()
        # adj_matrix = 1 - adj_matrix.to_dense()
        # adj_matrix = torch.ge(adj_matrix, self.mom_mask_threshold).float()

        momentum, momentum_edge_scores = [], []
        # build subgraph based on edge_scores_logit and threshold
        cut_edges = data.batch[data.edge_index[0]].bincount().cumsum(0)
        for i in range(data.num_graphs):

            if data[i].num_nodes == 0 or data[i].edge_index.shape[0] != 2 or data[i].edge_index.shape[1] < 1:
                self.logger.debug(
                    f"The graph (evt: {data[i].evt_num.item()}, "
                    f"run:  {data[i].run_num.item()}) has no nodes or edges."
                )
                if split:
                    momentum.append(torch.zeros(0, 3, device=data.x.device))
                    momentum_edge_scores.append(torch.zeros(0, 1, device=data.x.device))
                continue

            start_idx = cut_edges[i - 1] if i != 0 else 0
            end_idx = cut_edges[i]
            i_edge_scores_logit = edge_scores_logit[start_idx:end_idx]

            # MomNet to predict momenta
            momenta, mom_edge_scores = self.mom(data[i].x, data[i].edge_attr, data[i].edge_index,
                                                i_edge_scores_logit.unsqueeze(-1))
            momentum.append(momenta)
            momentum_edge_scores.append(mom_edge_scores)

        if split:
            return momentum, edge_scores, momentum_edge_scores
        else:
            return (
                torch.cat(momentum, dim=0),
                edge_scores,
                torch.cat(momentum_edge_scores, dim=0)
            )


def build_model(**kwargs):
    return TrkTrans(**kwargs)
