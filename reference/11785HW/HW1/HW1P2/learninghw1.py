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

# DataLoader workers prepare batches in parallel with GPU computation.
# Start with a small number because the dataset is large and preloaded.
NUM_WORKERS = 4
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

        # Pad each utterance independently so a context window cannot cross
        # an utterance boundary.
        raw_X = np.load(X_dir, allow_pickle=True)
        self.X = [
            np.pad(
                utterance,
                ((self.K, self.K), (0, 0)),
                mode="constant",
                constant_values=0,
            )
            for utterance in raw_X
        ]
        self.Y = None if self.test else np.load(Y_dir, allow_pickle=True)
        self.lookup = []

        # Keep original frame indices.  The corresponding center in the
        # padded utterance is frame_id + self.K.
        for utterance_id, utterance in enumerate(self.X):
            original_length = utterance.shape[0] - 2 * self.K
            self.lookup.extend(
                (utterance_id, frame_id)
                for frame_id in range(original_length)
            )

    def __len__(self):
        return len(self.lookup)

    def __getitem__(self, index):
        utterance_id, frame_id = self.lookup[index]
        center = frame_id + self.K
        utterance = self.X[utterance_id]

        context = utterance[center - self.K : center + self.K + 1]
        features = torch.tensor(context.reshape(-1), dtype=self.data_type)

        if self.test:
            return features

        label = torch.tensor(self.Y[utterance_id][frame_id], dtype=torch.long)
        return features, label


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
        train_dataset = DatasetHW1(
            X_dir=self.train_X,
            Y_dir=self.train_Y,
            context_K=self.params.K,
            data_type=self.dtype,
        )
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.params.B,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

    def _load_valid(self):
        validation_dataset = DatasetHW1(
            X_dir=self.valid_X,
            Y_dir=self.valid_Y,
            context_K=self.params.K,
            data_type=self.dtype,
        )
        self.valid_loader = torch.utils.data.DataLoader(
            validation_dataset,
            batch_size=self.params.B,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

    def _load_test(self):
        test_dataset = DatasetHW1(
            X_dir=self.test_X,
            Y_dir=None,
            context_K=self.params.K,
            data_type=self.dtype,
        )
        self.test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.params.B,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

    def test(self):
        if self.test_loader is None:
            self._load_test()

        predictions = []
        with torch.cuda.device(self.device):
            with torch.no_grad():
                self.model.eval()
                for batch in self.test_loader:
                    # DatasetHW1 returns only x for the test set.  This also
                    # handles a tuple in case the dataset implementation changes.
                    features = batch[0] if isinstance(batch, (tuple, list)) else batch
                    logits = self.model(
                        features.to(self.device, non_blocking=True)
                    )
                    predictions.append(torch.argmax(logits, dim=1).cpu())

            predictions = torch.cat(predictions).numpy().astype(np.int64)
            ids = np.arange(predictions.shape[0], dtype=np.int64)
            submission = np.column_stack((ids, predictions))

            output_path = os.path.join(
                os.path.dirname(os.path.abspath(self.params.data_dir)),
                "submission.csv",
            )
        np.savetxt(
            output_path,
            submission,
            delimiter=",",
            header="id,label",
            comments="",
            fmt="%d",
        )
        print(f"Saved {len(predictions)} predictions to {output_path}")
        return output_path
