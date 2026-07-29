"""Model definitions for HW1P2.

The input is a context window of mel-spectrogram frames. Each frame has 40
features and the classifier must produce logits for 71 phoneme states.
"""

import torch.nn as nn

from utils.base import Model


N_FEATURES = 40
N_CLASSES = 71


class MLP(Model):
    """Starter model interface.

    TODO: choose the hidden dimensions and implement the forward network.
    The output must have shape (batch_size, 71) and contain logits, not
    softmax probabilities.
    """

    def __init__(self, K, dropout=0.0):
        super().__init__()
        self.K = K
        self.dropout = dropout
        self.input_size = N_FEATURES * (2 * K + 1)

        # TODO: define self.network, for example with nn.Sequential.
        # A minimal first experiment could be:
        # self.network = nn.Sequential(
        #     nn.Linear(self.input_size, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, N_CLASSES),
        # )
        self.network = None

    @property
    def input_dims(self):
        return [self.input_size]

    def forward(self, x):
        if self.network is None:
            raise NotImplementedError("Define self.network before training")
        return self.network(x)
