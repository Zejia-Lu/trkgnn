import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.95, gamma=3, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets, weight=None):
        # Caculate BCE without reduction
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        if weight is not None:
            bce = bce * weight  # shape: [N] or [N, C]
        
        pt = torch.exp(-bce)
        loss = self.alpha * (1 - pt) ** self.gamma * bce

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1, max_neg_samples=2048):
        super().__init__()
        self.temp = temperature
        self.max_neg_samples = max_neg_samples  # 最大负例抽样数

    def forward(self, edge_feats, labels):
        pos_mask = labels == 1
        neg_mask = labels == 0
        
        if not pos_mask.any() or not neg_mask.any():
            return torch.tensor(0.0, device=edge_feats.device)
        
        pos_feats = edge_feats[pos_mask]  # [P, hidden_dim]
        neg_feats = edge_feats[neg_mask]  # [N, hidden_dim]
        
        # 负例抽样：若N超过max_neg_samples，则随机抽取
        if neg_feats.shape[0] > self.max_neg_samples:
            idx = torch.randperm(neg_feats.shape[0], device=neg_feats.device)[:self.max_neg_samples]
            neg_feats = neg_feats[idx]  # [max_neg_samples, hidden_dim]
        
        # 正例内部相似度（同上，保持不变）
        pos_sim = F.cosine_similarity(pos_feats.unsqueeze(1), pos_feats.unsqueeze(0), dim=2)
        mask = ~torch.eye(pos_feats.shape[0], pos_feats.shape[0], device=pos_sim.device, dtype=bool)
        pos_sim = pos_sim[mask]
        pos_loss = -torch.log(torch.sigmoid(pos_sim / self.temp)).mean() if pos_sim.numel() > 0 else 0.0
        
        # 正负例相似度（使用抽样后的负例）
        cross_sim = F.cosine_similarity(pos_feats.unsqueeze(1), neg_feats.unsqueeze(0), dim=2)  # [P, S]，S=min(N, max_neg_samples)
        neg_loss = -torch.log(torch.sigmoid(-cross_sim / self.temp)).mean() if cross_sim.numel() > 0 else 0.0
        
        return (pos_loss + neg_loss) / 2


def focal_contrastive_combined_loss(inputs, targets, edge_feats, weight=None, focal_alpha=0.95, focal_gamma=3, contrastive_weight=0.1, temperature=0.1):
    # 计算Focal Loss（带weight）
    focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    focal_loss = focal(inputs, targets, weight=weight)  # 传入weight
    
    # 计算对比损失（一般不需要weight，若需要可类似处理）
    contrastive = ContrastiveLoss(temperature=temperature)
    contrast_loss = contrastive(edge_feats, targets)
    
    # 总损失
    return focal_loss + contrastive_weight * contrast_loss


# class FocalLoss_new(nn.Module):
#     def __init__(self, alpha=0.786, gamma=2, reduction='mean'):
#         super().__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.reduction = reduction

#     def forward(self, inputs, targets, weight=None):
#         bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
#         # 如果 weight 提供，使用它来加权 BCE
#         if weight is not None:
#             bce = bce * weight

#         pt = torch.exp(-bce)
#         alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
#         focal_weight = alpha_t * (1 - pt) ** self.gamma
#         loss = focal_weight * bce

#         if self.reduction == 'mean':
#             return loss.mean()
#         elif self.reduction == 'sum':
#             return loss.sum()
#         else:
#             return loss

def dice_loss(pred, target):
    pred = torch.sigmoid(pred)
    numerator = 2 * (pred * target).sum()
    denominator = pred.sum() + target.sum()
    return 1 - numerator / denominator

def combined_loss(pred, target, weight=None, alpha=0.5):
    # BCE with optional weight
    if weight is not None:
        bce = F.binary_cross_entropy_with_logits(pred, target, weight=weight)
    else:
        bce = F.binary_cross_entropy_with_logits(pred, target)

    # Dice loss
    dice = dice_loss(pred, target)

    return alpha * bce + (1 - alpha) * dice