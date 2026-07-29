"""Dataset and training hooks for HW1P2.

The data directory must contain train.npy, train_labels.npy, dev.npy,
dev_labels.npy, and test.npy. The utterances have variable numbers of
frames, with 40 features per frame.
"""

import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from utils.base import Learning, Params


NUM_WORKERS = 0
N_FEATURES = 40


class ParamsHW1(Params):
    def __init__(
        self,
        K=5,
        B=1024,
        lr=1e-3,
        max_epoch=20,
        is_double=False,
        data_dir="",
        dropout=0.0,
        device="cuda:0",
    ):
        super().__init__(
            B=B,
            lr=lr,
            max_epoch=max_epoch,
            is_double=is_double,
            data_dir=data_dir,
            device=device,
        )
        self.K = K
        self.dropout = dropout
        dtype_name = "double" if is_double else "float"
        self.str = f"k={K}b={B}d={dropout}lr={lr}_{dtype_name}_"

    def __str__(self):
        return self.str


class DatasetHW1(Dataset):
    """Map utterance/frame pairs to padded context windows.

    TODO: load the utterances, pad each utterance on both sides by K zero
    frames, build a lookup from global index to (utterance_id, frame_id), and
    return a flattened context window from __getitem__.
    """

    def __init__(self, X_dir, Y_dir, context_K, data_type=torch.float):
        super().__init__()
        self.K = context_K
        self.data_type = data_type
        self.test = Y_dir is None

        # TODO: implement data loading and padding.
        self.X = np.load(X_dir, allow_pickle=True)
        self.Y = None if self.test else np.load(Y_dir, allow_pickle=True)
        self.lookup = []

        raise NotImplementedError("Implement DatasetHW1 preprocessing")

    def __len__(self):
        return len(self.lookup)

    def __getitem__(self, index):
        raise NotImplementedError("Implement DatasetHW1.__getitem__")


class LearningHW1(Learning):
    """Data-loader and submission hooks used by utils.base.Learning."""

    def __init__(self, params: ParamsHW1, model):
        super().__init__(params, model, torch.optim.Adam, nn.CrossEntropyLoss)

        self.train_X = os.path.join(params.data_dir, "train.npy")
        self.train_Y = os.path.join(params.data_dir, "train_labels.npy")
        self.valid_X = os.path.join(params.data_dir, "dev.npy")
        self.valid_Y = os.path.join(params.data_dir, "dev_labels.npy")
        self.test_X = os.path.join(params.data_dir, "test.npy")
        self.dtype = torch.double if params.is_double else torch.float

    def _load_train(self):
        raise NotImplementedError("Create the training Dataset and DataLoader")

    def _load_valid(self):
        raise NotImplementedError("Create the validation Dataset and DataLoader")

    def _load_test(self):
        raise NotImplementedError("Create the test Dataset and DataLoader")

    def test(self):
        raise NotImplementedError("Write predictions in the Kaggle CSV format")
