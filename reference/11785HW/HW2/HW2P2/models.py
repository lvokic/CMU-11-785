import torch
from torch import nn

from utils.base import Model


class CNN(Model):
    """Starter CNN interface for HW2P2 classification/verification."""

    def __init__(self, params):
        super().__init__(params)

        # TODO: define the convolutional network and its classifier.
        self.network = None

    @property
    def input_dims(self):
        return list(self.params.input_dims)

    def forward(self, x: torch.Tensor):
        if self.network is None:
            raise NotImplementedError("Define CNN.network before training")
        return self.network(x)
