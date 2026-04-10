import torch
import torch.nn as nn
from config import ALPHA_ABO, BETA_RH, NUM_CLASSES_ABO, NUM_CLASSES_RH

class MultiTaskLoss(nn.Module):
    def __init__(self, abo_weights=None, rh_weights=None):
        """
        Multi-task loss for ABO and Rh classification.

        Args:
            abo_weights: Class weights for ABO classes
            rh_weights: Class weights for Rh classes
        """
        super(MultiTaskLoss, self).__init__()
        self.abo_criterion = nn.CrossEntropyLoss(weight=abo_weights)
        self.rh_criterion = nn.CrossEntropyLoss(weight=rh_weights)
        self.alpha = ALPHA_ABO
        self.beta = BETA_RH

    def forward(self, abo_logits, rh_logits, abo_targets, rh_targets):
        """
        Compute multi-task loss.

        Args:
            abo_logits: ABO predictions (batch_size, 4)
            rh_logits: Rh predictions (batch_size, 2)
            abo_targets: ABO ground truth (batch_size,)
            rh_targets: Rh ground truth (batch_size,)

        Returns:
            total_loss: Weighted sum of losses
            abo_loss: ABO loss
            rh_loss: Rh loss
        """
        abo_loss = self.abo_criterion(abo_logits, abo_targets)
        rh_loss = self.rh_criterion(rh_logits, rh_targets)
        total_loss = self.alpha * abo_loss + self.beta * rh_loss

        return total_loss, abo_loss, rh_loss