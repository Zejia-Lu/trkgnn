import torch
import torch.nn as nn
from torch_geometric.nn.pool.graclus import graclus
from torch_geometric.data import Data
from torch_scatter import scatter_mean

from .MomNet import MomNet
from .utils import make_mlp


class HierarchicalGNN(nn.Module):
    def __init__(
            self, node_input_dim, edge_input_dim, hidden_dim, num_clusters, heads=4, n_encoder_layers=2,
            n_iterations=1
    ):
        super(HierarchicalGNN, self).__init__()

        self.num_clusters = num_clusters

        # Low-level network
        self.low_level_net = MomNet(node_input_dim, edge_input_dim, hidden_dim, heads, n_encoder_layers, n_iterations)

        # Supernode classifier
        self.track_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, data):
        momentum_change, edge_scores = self.low_level_net(data)

        # Enhance node features with edge_scores and momentum_changes
        avg_edge_score = scatter_mean(edge_scores, data.edge_index[0], dim=0, dim_size=data.num_nodes)
        avg_momentum = scatter_mean(momentum_change, data.edge_index[0], dim=0, dim_size=data.num_nodes)
        enhanced_node_features = torch.cat([data.x, avg_edge_score.unsqueeze(-1), avg_momentum.unsqueeze(-1)], dim=1)

        # Assuming edge_scores and edge_momenta are your edge scores and momenta tensors respectively
        normalized_edge_scores = (edge_scores - edge_scores.min()) / (edge_scores.max() - edge_scores.min())
        normalized_momenta = (momentum_change - momentum_change.min()) / (momentum_change.max() - momentum_change.min())
        combined_weights = normalized_edge_scores * normalized_momenta

        # Cluster nodes into supernodes
        s = graclus(data.edge_index, combined_weights, num_nodes=data.num_nodes)

        # Supernode classification to determine tracks
        supernode_scores = torch.sigmoid(self.track_classifier(s))
        supernode_predictions = (supernode_scores > 0.5).float()

        # Assign edges to supernodes
        # edge_assignments = self.assign_edges_to_supernodes(data.edge_index, assignment_matrix)

        # return supernode_predictions, edge_assignments


def assign_edges_to_supernodes(edge_index, assignment_matrix):
    edge_assignments = []

    for i, j in edge_index.t():
        # Get the assignment vectors for nodes i and j
        node_i_assignment = assignment_matrix[i]
        node_j_assignment = assignment_matrix[j]

        # Combine the assignments
        combined_assignment = node_i_assignment * node_j_assignment

        # Determine the supernode with the highest combined value
        supernode = combined_assignment.argmax().item()

        edge_assignments.append(supernode)

    return torch.tensor(edge_assignments)
