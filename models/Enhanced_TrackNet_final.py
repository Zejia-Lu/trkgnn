import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv
from torch_scatter import scatter_add

from .utils import make_mlp
from utility.FunctionTime import timing_decorator

class CBAM(nn.Module):
    def __init__(self, dim, reduction_ratio=16):
        super().__init__()
        self.dim = dim
        self.reduction_ratio = reduction_ratio
        self.channel_attn = nn.Sequential(
            nn.Linear(2 * dim, dim // reduction_ratio),
            nn.ReLU(),
            nn.Linear(dim // reduction_ratio, dim),
            nn.Sigmoid()
        )
        self.spatial_attn = nn.Sequential(
            nn.Conv1d(2, 1, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(1, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    @timing_decorator
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        max_pool = torch.max(x, dim=0, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=0, keepdim=True)
        cat_pool = torch.cat([max_pool, avg_pool], dim=1)
        channel_weights = self.channel_attn(cat_pool)
        x = x * channel_weights.expand_as(x)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        max_pool_spatial = torch.max(x, dim=-1, keepdim=True)[0]
        avg_pool_spatial = torch.mean(x, dim=-1, keepdim=True)
        cat_pool_spatial = torch.cat([max_pool_spatial, avg_pool_spatial], dim=-1)
        cat_pool_spatial = cat_pool_spatial.permute(0, 2, 1)
        spatial_weights = self.spatial_attn(cat_pool_spatial)
        x = x * spatial_weights.permute(0, 2, 1).expand_as(x)
        return x.squeeze(0)

class GatedFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )

    @timing_decorator
    def forward(self, x1, x2):
        gate_weights = self.gate(torch.cat([x1, x2], dim=1))
        return x1 * gate_weights + x2 * (1 - gate_weights)


class Enhanced_TrackNet(nn.Module):
    def __init__(self, node_input_dim, edge_input_dim, hidden_dim, heads=8, n_iterations=5, z_values=None):
        super(Enhanced_TrackNet, self).__init__()
        self.n_iterations = n_iterations
        self.hidden_dim = hidden_dim
        self.node_embedding = make_mlp(node_input_dim, [hidden_dim], dropout=0.1)
        self.edge_embedding = make_mlp(edge_input_dim + 3, [hidden_dim], dropout=0.1)
        if z_values is None:
            self.z_values = torch.tensor([7.98, 22.98, 38.98, 53.98, 89.98, 180.48])
        else:
            self.z_values = torch.unique(torch.as_tensor(z_values, dtype=torch.float32))
        self.z_to_index = {int(round(z.item())): i for i, z in enumerate(self.z_values)}
        self.register_buffer('z_keys', torch.tensor(list(self.z_to_index.keys()), dtype=torch.long))
        self.register_buffer('z_indices', torch.tensor(list(self.z_to_index.values()), dtype=torch.long))
        self.z_embedding = nn.Embedding(len(self.z_values), hidden_dim)
        self.cbam_node = CBAM(hidden_dim)
        self.cbam_edge = CBAM(hidden_dim)
        # # Version 2
        # self.edge_attention_layers = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Linear(hidden_dim, hidden_dim // 2),  # 压缩维度
        #         nn.ReLU(),
        #         nn.Linear(hidden_dim // 2, 1)  # 输出每条边的权重分数
        #     ) for _ in range(n_iterations)  # 每轮迭代用独立的注意力层
        # ])

        self.transformer_conv_list = nn.ModuleList([
            TransformerConv(2 * hidden_dim, hidden_dim, heads=heads)
            for _ in range(n_iterations)
        ])
        self.norm_transconv_list = nn.ModuleList([
            nn.LayerNorm(heads * hidden_dim)
            for _ in range(n_iterations)
        ])
        self.norm_combined_list = nn.ModuleList([
            nn.LayerNorm(2 * hidden_dim)
            for _ in range(n_iterations)
        ])
        self.projection_layer_edge_list = nn.ModuleList([
            nn.Linear(heads * hidden_dim, hidden_dim)
            for _ in range(n_iterations)
        ])
        self.projection_layer_node_list = nn.ModuleList([
            nn.Linear(heads * hidden_dim, hidden_dim)
            for _ in range(n_iterations)
        ])
        self.gfm_list = nn.ModuleList([
            GatedFusion(hidden_dim) for _ in range(n_iterations)
        ])
        self.edge_classifier = make_mlp(2 * hidden_dim + edge_input_dim, [hidden_dim, 1], output_activation=None)

    @timing_decorator
    def forward(self, data):
        send_idx = torch.cat([data.edge_index[0], data.edge_index[1]], dim=0)
        recv_idx = torch.cat([data.edge_index[1], data.edge_index[0]], dim=0)
        edge_indices = torch.stack([send_idx, recv_idx], dim=0)
        edge_attr_bi = torch.cat([data.edge_attr, data.edge_attr], dim=0)
        src_pos = data.x[edge_indices[0]][:, :3]
        dst_pos = data.x[edge_indices[1]][:, :3]
        rel_pos = src_pos - dst_pos
        z_values = data.x[:, 2]
        z_integers = torch.round(z_values).long()
        valid_mask = torch.isin(z_integers, self.z_keys)
        if not valid_mask.all():
            invalid_z = z_integers[~valid_mask].unique()
        z_indices = torch.zeros_like(z_integers)
        for key, idx in self.z_to_index.items():
            z_indices[z_integers == key] = idx
        node_features = self.node_embedding(data.x) + self.z_embedding(z_indices)
        node_features = self.cbam_node(node_features)
        edge_features = torch.cat([edge_attr_bi, rel_pos], dim=1)
        edge_features = self.edge_embedding(edge_features)
        edge_features = self.cbam_edge(edge_features)
        for i in range(self.n_iterations):
            x0 = node_features
            e0 = edge_features

            # Version 1: 使用边特征直接聚合
            aggregated_from_src = scatter_add(
                edge_features, edge_indices[1],
                dim=0, dim_size=node_features.shape[0]
            )
            combined_features = torch.cat([node_features, aggregated_from_src - node_features], dim=1)

            # # Version 2: 使用注意力机制计算边的权重
            # edge_scores = self.edge_attention_layers[i](edge_features)  # [E, 1]，E为边数
            # edge_attention = F.softmax(edge_scores, dim=0)  # 归一化权重，总和为1
            # # 加权聚合：重要的边（如正例边）贡献更大
            # weighted_edge_features = edge_features * edge_attention  # [E, hidden_dim]
            # aggregated_from_src = scatter_add(
            #     weighted_edge_features, 
            #     edge_indices[1], 
            #     dim=0, dim_size=node_features.shape[0]
            # )
            # combined_features = torch.cat([node_features, aggregated_from_src], dim=1)  # [N, 2*hidden_dim]

            
            combined_features = self.norm_combined_list[i](combined_features)
            trans_out = self.transformer_conv_list[i](combined_features, edge_indices)
            trans_out = self.norm_transconv_list[i](trans_out)
            node_proj = self.projection_layer_node_list[i](trans_out)
            # Version 1
            edge_proj = self.projection_layer_edge_list[i](trans_out[edge_indices[0]])

            # # Version 2
            # src_node_feat = trans_out[edge_indices[0]]  # 源节点特征
            # dst_node_feat = trans_out[edge_indices[1]]  # 目标节点特征
            # edge_proj = self.projection_layer_edge_list[i](torch.cat([src_node_feat, dst_node_feat], dim=1))

            node_features = self.gfm_list[i](x0, node_proj)
            edge_features = self.gfm_list[i](e0, edge_proj)
        start_idx, end_idx = data.edge_index
        clf_input = torch.cat([node_features[start_idx], node_features[end_idx], data.edge_attr], dim=1)
        edge_output = self.edge_classifier(clf_input).squeeze(-1)

        # Version 3
        original_edge_features = edge_features[:data.edge_index.shape[1]]  # 取前半部分（对应原始边）
        return edge_output, original_edge_features
        # return edge_output

class TrkTrans(nn.Module):
    def __init__(self, node_input_dim, edge_input_dim, hidden_dim, heads=8, n_iterations=5, z_values=None):
        super(TrkTrans, self).__init__()
        self.link = Enhanced_TrackNet(
            node_input_dim=node_input_dim,
            edge_input_dim=edge_input_dim,
            hidden_dim=hidden_dim,
            heads=heads,
            n_iterations=n_iterations,
            z_values=z_values
        )

    @timing_decorator
    def forward(self, data):
        # edge_scores = self.link(data)
        # return edge_scores
        
        # Version 3
        edge_scores, edge_feats = self.link(data)
        return edge_scores, edge_feats

def build_model(**kwargs):
    return TrkTrans(**kwargs)