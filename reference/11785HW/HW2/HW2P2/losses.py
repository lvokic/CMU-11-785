import torch
from torch import nn


class CenterLoss(nn.Module):
    """Starter for the optional center-loss component."""

    def __init__(self, params):
        super().__init__()
        self.params = params

    def forward(self, features, labels):
        raise NotImplemented


class CrossEntropyCenterLoss(nn.Module):
    """Starter for combining classification and center loss."""

    def __init__(self, params):
        super().__init__()
        self.params = params

    def forward(self, features, logits, labels):
        raise NotImplemented
