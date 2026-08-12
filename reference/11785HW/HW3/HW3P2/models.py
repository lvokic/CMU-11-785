from utils.base import *
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class ModelHW3(Model):
    def __init__(self, params):
        super().__init__(params)
        dropout = params.dropout

        # Preserve the time dimension; only reduce the frequency dimension.
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2)),  # 40 -> 20 frequency bins
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2)),  # 20 -> 10 frequency bins
        )
        self.feature_projection = nn.Sequential(
            nn.Linear(128 * 10, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.encoder = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.classifier = nn.Linear(512, params.output_channels)

    def forward(self, x: torch.Tensor, lengths: tuple):
        """

        :param x: padded (B,T,C)
        :param lengths:
        :return:
        """
        # (B, T, 40) -> (B, 1, T, 40)
        x = x.unsqueeze(1)
        x = self.cnn(x)  # (B, 128, T, 10)

        # Each time step becomes one 1280-dimensional CNN feature vector.
        batch_size, channels, time_steps, frequency_bins = x.shape
        x = x.permute(0, 2, 1, 3).reshape(
            batch_size, time_steps, channels * frequency_bins
        )
        x = self.feature_projection(x)

        packed = pack_padded_sequence(
            x,
            torch.as_tensor(lengths, dtype=torch.long, device="cpu"),
            batch_first=True,
            enforce_sorted=False,
        )
        packed, _ = self.encoder(packed)
        x, _ = pad_packed_sequence(
            packed, batch_first=True, total_length=time_steps
        )

        logits = self.classifier(x)  # (B, T, 42)
        return F.log_softmax(logits, dim=2).transpose(0, 1)  # (T, B, 42)
