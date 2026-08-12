"""HW3P2 — Utterance-to-phoneme mapping (clean handout entry point).

The Spring 2021 writeup intentionally did not prescribe a training framework.
Implement the data pipeline, training loop, decoding, and CSV generation here.

Data directory expected by the writeup:
    train.npy, train_labels.npy, dev.npy, dev_labels.npy, test.npy,
    sample_submission.csv, phoneme_list.py
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchaudio.models.decoder import cuda_ctc_decoder
from tqdm import tqdm

from models import ModelHW3
from utils.base import Learning, Params
from utils.phoneme_list import N_PHONEMES, PHONEME_MAP

N_FEATURES = 40
N_CLASSES = N_PHONEMES + 1  # CTC blank is class 0.


class ParamsHW3(Params):
    """HW3P2-specific parameters consumed by ModelHW3 and Learning."""

    def __init__(self, batch_size, lr, dropout, device, max_epoch=20, data_dir=""):
        super().__init__(
            B=batch_size,
            lr=lr,
            dropout=dropout,
            max_epoch=max_epoch,
            data_dir=str(data_dir),
            device=torch.device(device),
            input_dims=(N_FEATURES,),
            output_channels=N_CLASSES,
        )

    def __str__(self):
        return f"b={self.B}_lr={self.lr}_dropout={self.dropout}"


class PhonemeDataset(Dataset):
    """Load variable-length mel features and optional phoneme targets."""

    def __init__(self, x_path: Path, y_path: Path | None = None):
        self.features = np.load(x_path, allow_pickle=True)
        self.targets = None if y_path is None else np.load(y_path, allow_pickle=True)

        if self.targets is not None:
            assert len(self.features) == len(self.targets)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int):
        x = torch.as_tensor(self.features[index], dtype=torch.float32)
        if self.targets is None:
            return x
        y = torch.as_tensor(self.targets[index], dtype=torch.long)
        return x, y


def collate_train(batch):
    """TODO: pad a feature/label batch and return CTC-compatible lengths."""
    features, targets = zip(*batch)
    input_lengths = torch.tensor([x.shape[0] for x in features], dtype=torch.long)
    target_lengths = torch.tensor([x.shape[0] for x in targets], dtype=torch.long)
    padded_features = pad_sequence(list(features), batch_first=True, padding_value=0.0)
    padded_targets = pad_sequence(list(targets), batch_first=True, padding_value=0)
    return padded_features, padded_targets, input_lengths, target_lengths


def collate_test(batch):
    """TODO: pad a feature-only batch and return feature lengths."""
    feature_lengths = torch.tensor([x.shape[0] for x in batch], dtype=torch.long)
    padded_features = pad_sequence(batch, batch_first=True, padding_value=0.0)
    return padded_features, feature_lengths


def edit_distance(predicted, target):
    """Levenshtein distance between two phoneme-ID sequences."""
    previous = list(range(len(target) + 1))
    for i, predicted_token in enumerate(predicted, start=1):
        current = [i]
        for j, target_token in enumerate(target, start=1):
            current.append(min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (predicted_token != target_token),
            ))
        previous = current
    return previous[-1]


class HW3(Learning):
    """Required concrete shell around the abstract ``Learning`` base class."""

    def __init__(self, params: ParamsHW3, model: ModelHW3):
        super().__init__(params, model, torch.optim.Adam, nn.CTCLoss)
        self.decoder = cuda_ctc_decoder(
            tokens=PHONEME_MAP, nbest=1, beam_size=10
        )

    def _load_train(self):
        dataset = PhonemeDataset(
            Path(self.params.data_dir) / "train.npy",
            Path(self.params.data_dir) / "train_labels.npy",
        )
        self.train_loader = DataLoader(
            dataset,
            batch_size=self.params.B,
            shuffle=True,
            collate_fn=collate_train,
            num_workers=0,
        )

    def _load_valid(self):
        dataset = PhonemeDataset(
            Path(self.params.data_dir) / "dev.npy",
            Path(self.params.data_dir) / "dev_labels.npy",
        )
        self.valid_loader = DataLoader(
            dataset,
            batch_size=self.params.B,
            shuffle=False,
            collate_fn=collate_train,
            num_workers=0,
        )

    def _load_test(self):
        dataset = PhonemeDataset(Path(self.params.data_dir) / "test.npy")
        self.test_loader = DataLoader(
            dataset,
            batch_size=self.params.B,
            shuffle=False,
            collate_fn=collate_test,
            num_workers=0,
        )

    def train_one_epoch(self):
        if self.train_loader == None:
            self._load_train()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        with torch.cuda.device(self.device):
            self.model.train()
            for bx, by, input_lengths, target_lengths in tqdm(self.train_loader):
                bx = bx.to(self.device)
                by = by.to(self.device)

                log_probs = self.model(bx, input_lengths)
                loss = self.criterion(log_probs, by, input_lengths, target_lengths)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                batch_size = bx.shape[0]
                total_loss += loss.item() * batch_size
                total_samples += batch_size

                decoder_lengths = input_lengths.to(
                    device=self.device, dtype=torch.int32
                )
                hypotheses = self.decoder(
                    log_probs.detach().transpose(0, 1).contiguous(), decoder_lengths
                )

                for b, hypothesis_list in enumerate(hypotheses):
                    preidcted_ids = hypothesis_list[0].tokens.tolist()
                    true_ids = by[b, : target_lengths[b]].tolist()
                    if preidcted_ids == true_ids:
                        total_correct += 1

        mean_loss = total_loss / total_samples
        sequence_accuracy = total_correct / total_samples
        return mean_loss, sequence_accuracy

    def train(self, checkpoint_interval=5):
        if self.train_loader is None:
            self._load_train()

        Path("checkpoints").mkdir(exist_ok=True)
        print("Training...")
        with torch.cuda.device(self.device):
            self.model.train()
            for epoch in range(self.init_epoch + 1, self.params.max_epoch + 1):
                train_loss, train_acc = self.train_one_epoch()
                self.writer.add_scalar("Loss/Train", train_loss, epoch)
                self.writer.add_scalar("Accuracy/Train", train_acc, epoch)
                print(
                    f"Epoch {epoch}: "
                    f"train_loss={train_loss:.5f} "
                    f"sequence_acc={train_acc:.5f}"
                )

                val_loss, val_per = self.validate()
                self.writer.add_scalar("Loss/Validation", val_loss, epoch)
                self.writer.add_scalar("PER/Validation", val_per, epoch)
                print(
                    f"Epoch {epoch}: "
                    f"val_loss={val_loss:.5f} val_PER={val_per:.4%}"
                )
                self.model.train()
                if epoch % checkpoint_interval == 0:
                    self.save_model(epoch)

    def validate(self):
        if self.valid_loader is None:
            self._load_valid()

        total_loss = 0.0
        total_samples = 0
        total_edit_distance = 0
        total_target_phonemes = 0

        print("Validating...")
        with torch.cuda.device(self.device):
            with torch.no_grad():
                self.model.eval()
                for bx, by, input_lengths, target_lengths in tqdm(self.valid_loader):
                    bx = bx.to(self.device)
                    by = by.to(self.device)

                    log_probs = self.model(bx, input_lengths)
                    loss = self.criterion(log_probs, by, input_lengths, target_lengths)

                    batch_size = bx.shape[0]
                    total_loss += loss.item() * batch_size
                    total_samples += batch_size
                    decoder_lengths = input_lengths.to(
                        device=self.device, dtype=torch.int32
                    )
                    hypotheses = self.decoder(
                        log_probs.detach().transpose(0, 1).contiguous(), decoder_lengths
                    )

                    for b, hypothesis_list in enumerate(hypotheses):
                        predicted_ids = hypothesis_list[0].tokens.tolist()
                        target_length = int(target_lengths[b])
                        true_ids = by[b, :target_length].tolist()
                        total_edit_distance += edit_distance(predicted_ids, true_ids)
                        total_target_phonemes += target_length

        mean_loss = total_loss / total_samples
        phoneme_error_rate = total_edit_distance / total_target_phonemes
        return mean_loss, phoneme_error_rate

    def test(self):
        """TODO: decode test predictions and write submission.csv."""
        if self.test_loader is None:
            self._load_test()

        predictions = []
        print("Testing...")
        with torch.cuda.device(self.device):
            with torch.no_grad():
                self.model.eval()
                for bx, input_lengths in tqdm(self.test_loader):
                    bx = bx.to(self.device)

                    log_probs = self.model(bx, input_lengths)
                    decoder_lengths = input_lengths.to(
                        device=self.device, dtype=torch.int32
                    )
                    hypotheses = self.decoder(
                        log_probs.detach().transpose(0, 1).contiguous(), decoder_lengths
                    )
                    for hypothesis_list in hypotheses:
                        predicted_ids = hypothesis_list[0].tokens.tolist()
                        phoneme_string = "".join(
                            PHONEME_MAP[int(token)] for token in predicted_ids
                        )
                        predictions.append(phoneme_string)

        output_path = Path("submission.csv")
        with output_path.open("w", encoding="utf-8") as file:
            file.write("id,label\n")
            for index, prediction in enumerate(predictions):
                file.write(f"{index},{prediction}\n")
        print(f"Saved {len(predictions)} predictions to {output_path}")
        return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--save_every", type=int, default=5)
    args = parser.parse_args()

    # TODO: construct ModelHW3, loaders, Adam/AdamW, and nn.CTCLoss(blank=0).
    # TODO: train, validate, decode the test set, and call write_submission().
    if not args.train and not args.test:
        parser.error("Specify at least one action: --train and/or --test")
    params = ParamsHW3(
        batch_size=args.batch_size,
        lr=args.lr,
        dropout=args.dropout,
        device=args.device,
        max_epoch=args.epochs,
        data_dir=args.data_dir,
    )
    model = ModelHW3(params)
    learner = HW3(params, model)

    if args.train:
        learner.train(checkpoint_interval=args.save_every)
    if args.test:
        learner.test()


if __name__ == "__main__":
    main()
