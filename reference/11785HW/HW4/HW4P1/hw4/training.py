"""HW4P1 — WikiText-2 language modelling (clean handout reconstruction).

Implement the TODOs in this file.  The supplied fixtures and ``tests.py``
define the prediction and generation interfaces expected by the assignment.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


SEQ_LENGTH = 70


class LanguageModelSet(Dataset):
    """TODO: turn tokenized articles into input/next-token sequence pairs."""

    def __init__(self, data_loaded, sequence_length=SEQ_LENGTH):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class LanguageModelDataLoader(DataLoader):
    """TODO: provide shuffled language-model batches."""

    def __init__(self, dataset, batch_size, shuffle=True):
        raise NotImplementedError


class LockedDropOut(nn.Module):
    """TODO: apply one dropout mask across the time dimension."""

    def __init__(self, time_dim=0):
        super().__init__()
        self.time_dim = time_dim

    def forward(self, x, p):
        raise NotImplementedError


class LanguageModel(nn.Module):
    """TODO: embedding → recurrent layers → vocabulary logits.

    Implement weight tying and the recurrent regularization described in the
    writeup as you progress.  Return logits and any recurrent state needed for
    autoregressive generation.
    """

    def __init__(self, vocab_size):
        super().__init__()
        raise NotImplementedError

    def forward(self, x, hidden_state=None):
        raise NotImplementedError


class LanguageModelTrainer:
    """TODO: own the optimizer, loss, training, evaluation, and checkpoints."""

    def __init__(self, model, loader, max_epochs=1, run_id="exp"):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def train_batch(self, inputs, targets):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError


class TestLanguageModel:
    """Autolab-facing inference functions described in the writeup."""

    @staticmethod
    def prediction(inp, model):
        """TODO: return raw next-token logits of shape (batch, vocab)."""
        raise NotImplementedError

    @staticmethod
    def generation(inp, forward, model):
        """TODO: autoregressively generate ``forward`` token IDs."""
        raise NotImplementedError
