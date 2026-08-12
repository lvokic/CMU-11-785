"""HW4P2 — Listen, Attend and Spell (clean handout reconstruction)."""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from models import Model1


PAD_INDEX = 0  # Reuse as <eos> if desired, per the writeup.
N_FEATURES = 40


class ParamsHW4:
    """TODO: store model, optimization, teacher-forcing, and data parameters."""

    def __init__(self, **kwargs):
        for name, value in kwargs.items():
            setattr(self, name, value)


class DatasetHW4(Dataset):
    """TODO: load variable-length speech features and optional transcripts."""

    def __init__(self, x_path: Path, y_path: Path | None = None):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


def collate_train_val(batch):
    """TODO: right-pad utterances/transcripts and return both length tensors."""
    raise NotImplementedError


def collate_test(batch):
    """TODO: right-pad utterances and return input lengths."""
    raise NotImplementedError


class HW4:
    """TODO: LAS training/validation/test orchestration.

    Implement data loading, masked cross entropy, teacher forcing scheduling,
    validation edit distance, checkpointing, and Kaggle CSV creation.
    """

    def __init__(self, params, model):
        raise NotImplementedError

    def _load_train(self):
        raise NotImplementedError

    def _load_valid(self):
        raise NotImplementedError

    def _load_test(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def validate(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    # TODO: preprocess transcripts; create ParamsHW4, Model1, and HW4.
    raise NotImplementedError("Implement the LAS pipeline before running HW4P2.")


if __name__ == "__main__":
    main()
