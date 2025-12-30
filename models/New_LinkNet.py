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

        # 通道注意力分支
        self.channel_attn = nn.Sequential(
            nn.Linear(2 * dim, dim // reduction_ratio),
            nn.ReLU(),
            nn.Linear(dim // reduction_ratio, dim),
            nn.Sigmoid()
        )

        # 空间注意力分支（适用于序列数据）
        self.spatial_attn = nn.Sequential(
            nn.Conv1d(2, 1, kernel_size=5, padding=2),  # 更小的卷积核
            nn.ReLU(),
            nn.Conv1d(1, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    @timing_decorator
    def forward(self, x):
        # 处理输入维度（确保至少2维：[N, C]）
        if x.dim() == 1:
            x = x.unsqueeze(0)  # [1, C] -> [N=1, C]

        # 通道注意力
        max_pool = torch.max(x, dim=0, keepdim=True)[0]  # [1, C]
        avg_pool = torch.mean(x, dim=0, keepdim=True)    # [1, C]
        cat_pool = torch.cat([max_pool, avg_pool], dim=1)  # [1, 2C]
        channel_weights = self.channel_attn(cat_pool)      # [1, C]
        x = x * channel_weights.expand_as(x)               # [N, C]

        # 空间注意力
        if x.dim() == 2:
            x = x.unsqueeze(0)  # [1, N, C]（添加batch维度）
        
        max_pool_spatial = torch.max(x, dim=-1, keepdim=True)[0]  # [1, N, 1]
        avg_pool_spatial = torch.mean(x, dim=-1, keepdim=True)     # [1, N, 1]
        cat_pool_spatial = torch.cat([max_pool_spatial, avg_pool_spatial], dim=-1)  # [1, N, 2]

        cat_pool_spatial = cat_pool_spatial.permute(0, 2, 1)  # [1, 2, N]
        spatial_weights = self.spatial_attn(cat_pool_spatial)  # [1, 1, N]
        x = x * spatial_weights.permute(0, 2, 1).expand_as(x)  # [1, N, C]

        return x.squeeze(0)  # 移除batch维度，返回[N, C]


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


class New_LinkNet(nn.Module):
    def __init__(self, node_input_dim, edge_input_dim, hidden_dim, heads=4, n_iterations=3, z_values=None):
        super(New_LinkNet, self).__init__()
        self.n_iterations = n_iterations
        self.hidden_dim = hidden_dim

        # 节点和边的嵌入层
        self.node_embedding = make_mlp(node_input_dim, [hidden_dim], dropout=0.1)
        self.edge_embedding = make_mlp(edge_input_dim + 3, [hidden_dim], dropout=0.1)  # +3为Δx,Δy,Δz

        # Z值处理（自动识别并初始化）
        if z_values is None:
            # 默认Z值（保留原代码中的值，但实际应从数据中获取）
            self.z_values = torch.tensor([7.98, 22.98, 38.98, 53.98, 89.98, 180.48])
        else:
            # 从输入Z值中提取唯一值
            self.z_values = torch.unique(torch.as_tensor(z_values, dtype=torch.float32))
        
        # Z值到索引的映射（取整后映射）
        self.z_to_index = {int(round(z.item())): i for i, z in enumerate(self.z_values)}
        self.register_buffer('z_keys', torch.tensor(list(self.z_to_index.keys()), dtype=torch.long))
        self.register_buffer('z_indices', torch.tensor(list(self.z_to_index.values()), dtype=torch.long))

        # Z值嵌入层
        self.z_embedding = nn.Embedding(len(self.z_values), hidden_dim)

        # 注意力模块
        self.cbam_node = CBAM(hidden_dim)
        self.cbam_edge = CBAM(hidden_dim)

        # Transformer卷积层
        self.transformer_conv_list = nn.ModuleList([
            TransformerConv(2 * hidden_dim, hidden_dim, heads=heads)  # 输入为2*hidden_dim（拼接后）
            for _ in range(n_iterations)
        ])

        # 归一化和投影层
        self.norm_transconv_list = nn.ModuleList([
            nn.LayerNorm(heads * hidden_dim)  # Transformer输出为heads*hidden_dim
            for _ in range(n_iterations)
        ])
        self.norm_combined_list = nn.ModuleList([
            nn.LayerNorm(2 * hidden_dim)  # 拼接后的特征归一化
            for _ in range(n_iterations)
        ])

        self.projection_layer_edge_list = nn.ModuleList([
            nn.Linear(heads * hidden_dim, hidden_dim)  # 边特征投影回hidden_dim
            for _ in range(n_iterations)
        ])
        self.projection_layer_node_list = nn.ModuleList([
            nn.Linear(heads * hidden_dim, hidden_dim)  # 节点特征投影回hidden_dim
            for _ in range(n_iterations)
        ])

        # 门控融合模块
        self.gfm_list = nn.ModuleList([
            GatedFusion(hidden_dim) for _ in range(n_iterations)
        ])

        # 边分类器
        self.edge_classifier = make_mlp(2 * hidden_dim, [hidden_dim, 1], output_activation=None)

    @timing_decorator
    def forward(self, data):
        # 处理双向边（确保无向性）
        send_idx = torch.cat([data.edge_index[0], data.edge_index[1]], dim=0)
        recv_idx = torch.cat([data.edge_index[1], data.edge_index[0]], dim=0)
        edge_indices = torch.stack([send_idx, recv_idx], dim=0)  # 双向边索引
        edge_attr_bi = torch.cat([data.edge_attr, data.edge_attr], dim=0)  # 双向边属性

        # 计算相对位置（假设data.x前3列为坐标）
        src_pos = data.x[edge_indices[0]][:, :3]  # 源节点坐标
        dst_pos = data.x[edge_indices[1]][:, :3]  # 目标节点坐标
        rel_pos = src_pos - dst_pos  # 相对位置差

        # 处理Z值（第3列，索引为2）
        z_values = data.x[:, 2]  # 提取Z坐标
        z_integers = torch.round(z_values).long()  # 四舍五入为整数

        # 检查是否有无效Z值
        valid_mask = torch.isin(z_integers, self.z_keys)
        if not valid_mask.all():
            invalid_z = z_integers[~valid_mask].unique()
            # 自动添加新Z值（可选）

        # 映射Z值到索引（向量化操作）
        z_indices = torch.zeros_like(z_integers)
        for key, idx in self.z_to_index.items():
            z_indices[z_integers == key] = idx

        # 节点特征：嵌入+Z值嵌入，再经CBAM
        node_features = self.node_embedding(data.x) + self.z_embedding(z_indices)
        node_features = self.cbam_node(node_features)

        # 边特征：属性+相对位置，嵌入后经CBAM
        edge_features = torch.cat([edge_attr_bi, rel_pos], dim=1)
        edge_features = self.edge_embedding(edge_features)
        edge_features = self.cbam_edge(edge_features)

        # 多轮Transformer迭代
        for i in range(self.n_iterations):
            # 保存当前特征用于残差连接
            x0 = node_features
            e0 = edge_features

            # 聚合边特征到节点
            aggregated_from_src = scatter_add(
                edge_features, edge_indices[1],  # 按目标节点聚合
                dim=0, dim_size=node_features.shape[0]
            )
            # 拼接节点特征和聚合特征（2*hidden_dim）
            combined_features = torch.cat([node_features, aggregated_from_src - node_features], dim=1)
            combined_features = self.norm_combined_list[i](combined_features)

            # Transformer卷积
            trans_out = self.transformer_conv_list[i](combined_features, edge_indices)
            trans_out = self.norm_transconv_list[i](trans_out)  # 归一化

            # 投影回hidden_dim
            node_proj = self.projection_layer_node_list[i](trans_out)
            edge_proj = self.projection_layer_edge_list[i](trans_out[edge_indices[0]])  # 边特征从源节点投影

            # 门控融合（残差连接）
            node_features = self.gfm_list[i](x0, node_proj)
            edge_features = self.gfm_list[i](e0, edge_proj)

        # 边分类：使用原始边（非双向）的节点特征拼接
        start_idx, end_idx = data.edge_index  # 原始单向边
        clf_input = torch.cat([node_features[start_idx], node_features[end_idx]], dim=1)
        edge_output = self.edge_classifier(clf_input).squeeze(-1)

        return edge_output


class TrkTrans(nn.Module):
    def __init__(self, node_input_dim, edge_input_dim, hidden_dim, heads=4, n_iterations=3, z_values=None):
        super(TrkTrans, self).__init__()
        self.link = New_LinkNet(
            node_input_dim=node_input_dim,
            edge_input_dim=edge_input_dim,
            hidden_dim=hidden_dim,
            heads=heads,
            n_iterations=n_iterations,
            z_values=z_values  # 传递Z值参数
        )

    @timing_decorator
    def forward(self, data):
        edge_scores = self.link(data)
        return edge_scores


def build_model(**kwargs):
    return TrkTrans(**kwargs)