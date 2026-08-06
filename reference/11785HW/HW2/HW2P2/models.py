import torch
from torch import nn

from utils.base import Model


class CNN(Model):
    """Starter CNN interface for HW2P2 classification/verification."""

    def __init__(self, params):
        super().__init__(params)

        in_channels = params.input_dims[0]
        num_classes = params.output_channels
        dropout = params.dropout
        self.embedding_dim = 512
        self.network = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, self.embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(self.embedding_dim, num_classes),
        )

    @property
    def input_dims(self):
        return list(self.params.input_dims)

    def forward(self, x: torch.Tensor):
        if self.network is None:
            raise NotImplementedError("Define CNN.network before training")
        embedding = self.encode(x)
        return self.network[-1](embedding)

    def encode(self, x: torch.Tensor):
        """Return the fixed-length embedding before the final classifier."""
        for layer in list(self.network.children())[:-1]:
            x = layer(x)
        return x
