import numpy as np
import torch
from sklearn.metrics import roc_auc_score


class EpochMetrics:
    def __init__(self, epoch, task_type):
        self.metrics = {
            'epoch': epoch,
            'train_loss': 0,
            'valid_loss': 0,
            'train_batches': 0,
            'valid_batches': 0,
        }

        self.task_type = task_type
        if task_type == 'link':
            self.valid_y_pred = np.empty((0))
            self.valid_y_true = np.empty(0)
            self.valid_y_weight = np.empty(0)
            self.valid_y_TP = 0
            self.valid_y_FP = 0
            self.valid_y_TN = 0
            self.valid_y_FN = 0
        elif task_type == 'momentum':
            self.valid_p_diff_truth = np.empty(0)
            self.valid_p_diff_fake = np.empty(0)

    def update_loss(self, loss: float, batch_size: int, stage: str = 'train'):
        self.metrics[f'{stage}_loss'] += loss
        self.metrics[f'{stage}_batches'] += batch_size

    def update_link(
            self,
            y_pred: torch.Tensor, y_true: torch.Tensor, y_weight: torch.Tensor, y_score: np.array,
    ):

        self.valid_y_pred = np.concatenate((self.valid_y_pred, y_score), axis=0)
        self.valid_y_true = np.concatenate((self.valid_y_true, y_true.detach().cpu().numpy()), axis=0).astype(int)
        self.valid_y_weight = np.concatenate((self.valid_y_weight, y_weight.detach().cpu().numpy()), axis=0)

        self.valid_y_TP += ((y_pred == 1) & (y_true == 1)).float().mul(y_weight).sum().item()
        self.valid_y_FP += ((y_pred == 1) & (y_true == 0)).float().mul(y_weight).sum().item()
        self.valid_y_TN += ((y_pred == 0) & (y_true == 0)).float().mul(y_weight).sum().item()
        self.valid_y_FN += ((y_pred == 0) & (y_true == 1)).float().mul(y_weight).sum().item()

    def update_momentum(self, p_diff: torch.Tensor, finite_mask: torch.Tensor, truth_mask: torch.Tensor):
        truth = p_diff[truth_mask & finite_mask]
        fake = p_diff[~truth_mask & finite_mask]

        self.valid_p_diff_truth = np.concatenate((self.valid_p_diff_truth, truth.detach().cpu().numpy()), axis=0)
        self.valid_p_diff_fake = np.concatenate((self.valid_p_diff_fake, fake.detach().cpu().numpy()), axis=0)

    def to_dict(self):

        self.metrics['train_loss'] /= self.metrics['train_batches']
        self.metrics['valid_loss'] /= self.metrics['valid_batches']

        del self.metrics['train_batches'], self.metrics['valid_batches']

        if self.task_type == 'link':
            y_correct = self.valid_y_TP + self.valid_y_TN
            y_sum = self.valid_y_TP + self.valid_y_FP + self.valid_y_TN + self.valid_y_FN
            self.metrics['valid_y_acc'] = y_correct / y_sum
            self.metrics['valid_y_precision'] = self.valid_y_TP / (self.valid_y_TP + self.valid_y_FP)
            self.metrics['valid_y_recall'] = self.valid_y_TP / (self.valid_y_TP + self.valid_y_FN)

            self.metrics['valid_y_auc'] = roc_auc_score(
                y_true=self.valid_y_true,
                y_score=self.valid_y_pred,
                sample_weight=self.valid_y_weight,
            )

        if self.task_type == 'momentum':
            self.metrics['valid_p_diff_truth_mean'] = np.mean(self.valid_p_diff_truth)
            self.metrics['valid_p_diff_truth_std'] = np.std(self.valid_p_diff_truth)

            self.metrics['valid_p_diff_fake_mean'] = np.mean(self.valid_p_diff_fake)
            self.metrics['valid_p_diff_fake_std'] = np.std(self.valid_p_diff_fake)

        return self.metrics
