import torch.nn as nn
import torch


class RelativeHuberLoss(nn.Module):
    def __init__(self, delta=1.0, epsilon=1e-6):
        super(RelativeHuberLoss, self).__init__()
        self.delta = delta
        self.epsilon = epsilon

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, weight: torch.Tensor = None):
        if weight is None:
            weight = torch.ones_like(y_true)

        y_zero = torch.abs(y_true) < self.epsilon

        # Compute the relative difference for non-zero y_true
        relative_diff = torch.where(~y_zero, (y_pred - y_true) / (y_true + self.epsilon), y_pred)

        # For y_true = 0, use absolute difference
        abs_diff = torch.abs(y_pred - y_true)
        relative_diff = torch.where(y_zero, abs_diff, relative_diff)

        abs_relative_diff = torch.abs(relative_diff)
        is_small_error = abs_relative_diff <= self.delta

        small_error_loss = 0.5 * (relative_diff ** 2)
        large_error_loss = self.delta * abs_relative_diff - 0.5 * (self.delta ** 2)

        # Apply sample weights
        loss = torch.where(is_small_error, small_error_loss, large_error_loss)
        weighted_loss = loss * weight

        return weighted_loss.mean()
