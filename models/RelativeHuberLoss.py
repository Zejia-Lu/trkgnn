import torch
import torch.nn as nn


class RelativeHuberLoss(nn.Module):
    def __init__(self, delta=1.0, epsilon=1e-6):
        super(RelativeHuberLoss, self).__init__()
        self.delta = delta
        self.epsilon = epsilon

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        relative_diff = (y_pred - y_true) / (y_true + self.epsilon)
        abs_relative_diff = torch.abs(relative_diff)

        is_small_error = abs_relative_diff <= self.delta

        small_error_loss = 0.5 * (relative_diff ** 2)
        large_error_loss = self.delta * abs_relative_diff - 0.5 * (self.delta ** 2)

        return torch.where(is_small_error, small_error_loss, large_error_loss).mean()
