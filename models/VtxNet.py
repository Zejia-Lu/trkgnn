# Externals
import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv, global_mean_pool

import logging

# Locals
from models.utils import make_mlp
from utility.FunctionTime import timing_decorator


class VtxNet(nn.Module):
    def __init__(self, node_input_dim, edge_input_dim, hidden_dim, heads=4, n_encoder_layers=2, n_iterations=1):
        super(VtxNet, self).__init__()

        self.n_iterations = n_iterations

        self.node_embedding = make_mlp(node_input_dim, [hidden_dim] * n_encoder_layers)
        self.edge_embedding = make_mlp(edge_input_dim, [hidden_dim] * n_encoder_layers)

        self.transformer_conv_list = nn.ModuleList([
            TransformerConv(hidden_dim, int(hidden_dim / heads), heads=heads, edge_dim=hidden_dim)
            for _ in range(n_iterations)
        ])

        self.norm_transconv_list = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(n_iterations)
        ])

        self.classifier = make_mlp(hidden_dim, [hidden_dim, 1], output_activation=None)
        self.regressor = make_mlp(hidden_dim, [hidden_dim, 1], output_activation=None)

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

        # Loop over iterations
        for i in range(self.n_iterations):
            x0 = node_features
            e0 = edge_features

            # Apply transformer conv
            node_features = self.transformer_conv_list[i](
                node_features,
                edge_indices,
                edge_attr=edge_features,
            )

            node_features = self.norm_transconv_list[i](node_features)

            edge_features = torch.cat([node_features[dst_indices] - node_features[src_indices]], dim=1)

            # shortcut
            node_features = node_features + x0
            edge_features = edge_features + e0

        # pooling
        global_feature = global_mean_pool(node_features, data.batch)

        # classification
        classification = self.classifier(global_feature).squeeze(-1)
        # regression
        regression = self.regressor(global_feature).squeeze(-1)

        return classification, regression


def build_model(**kwargs):
    return VtxNet(**kwargs)


if __name__ == '__main__':
    import torch
    from torch_geometric.data import Data
    import logging

    # Assuming the VtxNet class and build_model function are defined as provided

    # Create a simple logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()


    # Function to create a mock graph data
    def create_mock_data(node_count, edge_count, node_input_dim, edge_input_dim):
        x = torch.rand((node_count, node_input_dim))  # Node features
        edge_index = torch.randint(0, node_count, (2, edge_count))  # Edge indices
        edge_attr = torch.rand((edge_count, edge_input_dim))  # Edge attributes
        batch = torch.zeros(node_count, dtype=torch.long)  # Batch vector

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)


    # Test function
    def test_VtxNet():
        node_input_dim = 5
        edge_input_dim = 3
        hidden_dim = 16
        heads = 4
        n_encoder_layers = 2
        n_iterations = 1

        model = build_model(node_input_dim=node_input_dim, edge_input_dim=edge_input_dim, hidden_dim=hidden_dim,
                            heads=heads, n_encoder_layers=n_encoder_layers, n_iterations=n_iterations)
        logger.info("Model built successfully.")

        mock_data = create_mock_data(node_count=10, edge_count=20, node_input_dim=node_input_dim,
                                     edge_input_dim=edge_input_dim)
        logger.info("Mock data created successfully.")

        classification, regression = model(mock_data, verbose=True)
        logger.info(f"Classification output shape: {classification.shape}")
        logger.info(f"Regression output shape: {regression.shape}")


    # Run the test
    test_VtxNet()

    pass
