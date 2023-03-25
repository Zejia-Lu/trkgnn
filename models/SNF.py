import torch
import torch.nn as nn
import pyro.distributions as dist
from pyro.distributions.transforms import AffineCoupling
from models.mpnn import GNN
from models.utils import make_mlp


class SimpleCouplingNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleCouplingNetwork, self).__init__()
        self.mlp = make_mlp(input_dim, [hidden_dim] * 2, output_activation='Tanh')
        self.mean_layer = nn.Linear(hidden_dim, output_dim)
        self.log_scale_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        hidden = self.mlp(x)
        mean = self.mean_layer(hidden)
        log_scale = self.log_scale_layer(hidden)
        return mean, log_scale


class StochasticLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(StochasticLayer, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        mean = self.layers(x)
        log_std = torch.zeros_like(mean)
        noise = torch.randn_like(mean) * torch.exp(log_std)
        return mean + noise


def custom_split(x, nf_dim):
    x1, x2 = x.split([nf_dim, x.size(-1) - nf_dim], dim=-1)
    return x1, x2


class SNFModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, nf_dim=1, **kwargs):
        super(SNFModel, self).__init__()
        self.nf_dim = nf_dim
        self.gnn = GNN(input_dim, hidden_dim, snf_output_dim=nf_dim, **kwargs)
        self.stochastic_layer = StochasticLayer(nf_dim, hidden_dim)
        self.coupling_network = SimpleCouplingNetwork(nf_dim, hidden_dim, nf_dim)
        self.affine_coupling = AffineCoupling(nf_dim, self.coupling_network)

    def forward(self, data):
        # Pass input through GNN
        edge_scores, snf_par = self.gnn(data)

        # Pass GNN output through the Stochastic Layer
        stochastic_output = self.stochastic_layer(snf_par)

        # Pass the Stochastic Layer output through the SimpleCouplingNetwork
        mean, log_scale = self.coupling_network(stochastic_output)

        # Concatenate mean and log_scale along the last dimension
        coupled_output = torch.cat((mean, log_scale), dim=-1)

        # Pass the coupled output through the AffineCoupling
        flow_output = self.affine_coupling(coupled_output)

        return edge_scores, flow_output



def build_model(**kwargs):
    return SNFModel(**kwargs)
