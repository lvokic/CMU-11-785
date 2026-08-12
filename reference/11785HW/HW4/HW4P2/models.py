"""HW4P2 LAS model components (clean handout reconstruction)."""

import torch
import torch.nn as nn


class PBLSTM(nn.Module):
    """TODO: halve time resolution, then run a bidirectional LSTM."""

    def __init__(self, input_size, hidden_size):
        super().__init__()
        raise NotImplementedError

    def forward(self, x, lengths):
        raise NotImplementedError


class Encoder(nn.Module):
    """TODO: listener network yielding attention keys, values, and lengths."""

    def __init__(self, params):
        super().__init__()
        raise NotImplementedError

    def forward(self, x, lengths):
        raise NotImplementedError


class Attention(nn.Module):
    """TODO: score decoder query against encoder keys and produce context."""

    def __init__(self, key_size, query_size):
        super().__init__()
        raise NotImplementedError

    def forward(self, keys, values, query, lengths):
        raise NotImplementedError


class Decoder(nn.Module):
    """TODO: character embedding, attention-aware LSTMCell, and vocabulary logits."""

    def __init__(self, params):
        super().__init__()
        raise NotImplementedError

    def forward(self, keys, values, lengths, targets=None, teacher_forcing=0.0):
        raise NotImplementedError


class Model1(nn.Module):
    """TODO: combine listener encoder and attention-based speller decoder."""

    def __init__(self, params):
        super().__init__()
        raise NotImplementedError

    def forward(self, x, lengths, targets=None, teacher_forcing=0.0):
        raise NotImplementedError
