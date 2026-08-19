"""HW4P2 — Listen, Attend and Spell (clean handout reconstruction)."""

import argparse
import math
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Subset

from models import Model1
from utils.search import beam_search_batch
from utils.vocab import CharVocab
from utils.vocab import EOS_TOKEN, PAD_TOKEN, SOS_TOKEN
from tqdm import tqdm

PAD_INDEX = 0  # Reuse as <eos> if desired, per the writeup.
N_FEATURES = 40


@dataclass
class ParamsHW4:
    data_dir: Path
    device: str
    vocab_size: int
    pad_index: int
    sos_index: int

    feature_dim: int = 40
    encoder_hidden: int = 256
    attention_dim: int = 128
    embedding_size: int = 256
    decoder_hidden: int = 512
    max_decode_steps: int = 600

    dropout: float = 0.1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    # Keep the global batch modest: LAS benefits more from frequent optimizer
    # updates than from a very large DataParallel batch.
    batch_size: int = 128
    num_workers: int = 4
    epochs: int = 10
    teacher_forcing: float = 0.9
    # Keep teacher forcing fixed for the baseline.  A later experiment can
    # lower this value after acoustic alignment is reliable.
    teacher_forcing_end: float = 0.9
    teacher_forcing_decay_epochs: int = 12
    teacher_forcing_decay_start_epoch: int = 0
    amp: bool = True
    gpu_ids: tuple[int, ...] = ()
    decode_strategy: str = "greedy"
    beam_width: int = 5
    beam_length_penalty: float = 0.0
    max_test_samples: int | None = None

    checkpoint_dir: Path = Path("checkpoints/hw4p2")
    submission_path: Path = Path("submission.csv")


class DatasetHW4(Dataset):
    """TODO: load variable-length speech features and optional transcripts."""

    def __init__(
        self, x_path: Path, y_path: Path | None = None, vocab: CharVocab | None = None
    ):
        self.features = np.load(x_path, allow_pickle=True)
        self.vocab = vocab
        if y_path is None:
            self.targets = None
        else:
            if vocab is None:
                raise ValueError("A vocabulary is required for labeled data.")
            transcripts = np.load(y_path, allow_pickle=True)
            self.targets = [
                torch.tensor(
                    vocab.encode(vocab.transcript_to_text(transcript)),
                    dtype=torch.long,
                )
                for transcript in transcripts
            ]

    def __getitem__(self, index):
        feature = torch.as_tensor(
            np.asanyarray(self.features[index], dtype=np.float32), dtype=torch.float32
        )

        # Cepstral mean and variance normalization (CMVN): normalize every
        # acoustic feature dimension over the frames of this utterance.
        feature = (feature - feature.mean(dim=0, keepdim=True)) / feature.std(
            dim=0, keepdim=True, unbiased=False
        ).clamp_min(1e-5)

        if self.targets is None:
            return feature
        return feature, self.targets[index]

    def __len__(self):
        return len(self.features)


def collate_train_val(batch):
    """TODO: right-pad utterances/transcripts and return both length tensors."""
    features, targets = zip(*batch)
    input_lengths = torch.tensor(
        [feature.shape[0] for feature in features], dtype=torch.long
    )
    target_lenths = torch.tensor(
        [target.shape[0] for target in targets], dtype=torch.long
    )
    padded_features = pad_sequence(list(features), batch_first=True, padding_value=0.0)
    padded_targets = pad_sequence(
        list(targets), batch_first=True, padding_value=PAD_INDEX
    )
    return padded_features, padded_targets, input_lengths, target_lenths


def collate_test(batch):
    """Right-pad utterances and return input lengths."""
    input_lengths = torch.tensor(
        [feature.shape[0] for feature in batch],
        dtype=torch.long,
    )
    padded_features = pad_sequence(
        list(batch),
        batch_first=True,
        padding_value=0.0,
    )
    return padded_features, input_lengths


class HW4:
    """TODO: LAS training/validation/test orchestration.

    Implement data loading, masked cross entropy, teacher forcing scheduling,
    validation edit distance, checkpointing, and Kaggle CSV creation.
    """

    def __init__(self, params, model, vocab):
        self.params = params
        self.vocab = vocab
        if params.gpu_ids:
            if not torch.cuda.is_available():
                raise RuntimeError("--gpus was requested, but CUDA is unavailable.")
            self.device = torch.device(f"cuda:{params.gpu_ids[0]}")
        else:
            self.device = torch.device(params.device)
        self.model = model.to(self.device)
        if len(params.gpu_ids) > 1:
            self.model = nn.DataParallel(
                self.model,
                device_ids=list(params.gpu_ids),
                output_device=params.gpu_ids[0],
            )
        self.use_amp = bool(params.amp and self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=params.learning_rate,
            weight_decay=getattr(params, "weight_decay", 1e-4),
        )
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=params.pad_index,
            reduction="none",
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=2,
        )

        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None

        self.current_epoch = 0
        self.best_val_loss = float("inf")
        # The competition metric is free-running edit distance, not
        # teacher-forced loss.  Keep both so that optimization diagnostics and
        # checkpoint selection do not get conflated.
        self.best_val_cer = float("inf")

        self.checkpoint_dir = Path(getattr(params, "checkpoint_dir", "checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _load_train(self):
        train_set = DatasetHW4(
            x_path=self.params.data_dir / "train.npy",
            y_path=self.params.data_dir / "train_transcripts.npy",
            vocab=self.vocab,
        )
        self.train_loader = DataLoader(
            train_set,
            batch_size=self.params.batch_size,
            shuffle=True,
            num_workers=getattr(self.params, "num_workers", 4),
            pin_memory=(self.device.type == "cuda"),
            collate_fn=collate_train_val,
        )

    def _load_valid(self):
        valid_set = DatasetHW4(
            x_path=self.params.data_dir / "dev.npy",
            y_path=self.params.data_dir / "dev_transcripts.npy",
            vocab=self.vocab,
        )
        self.valid_loader = DataLoader(
            valid_set,
            batch_size=self.params.batch_size,
            shuffle=False,
            num_workers=getattr(self.params, "num_workers", 4),
            pin_memory=(self.device.type == "cuda"),
            collate_fn=collate_train_val,
        )

    def _load_test(self):
        test_set = DatasetHW4(
            x_path=self.params.data_dir / "test.npy",
        )
        if self.params.max_test_samples is not None:
            test_set = Subset(
                test_set,
                range(min(self.params.max_test_samples, len(test_set))),
            )
        self.test_loader = DataLoader(
            test_set,
            # LAS beam search retains several recurrent hypotheses per
            # utterance.  A moderate acoustic batch avoids a large padded
            # encoder allocation while the search itself is sequential over
            # utterances.
            batch_size=(
                min(self.params.batch_size, 32)
                if self.params.decode_strategy == "beam"
                else self.params.batch_size
            ),
            shuffle=False,
            num_workers=getattr(self.params, "num_workers", 4),
            pin_memory=(self.device.type == "cuda"),
            collate_fn=collate_test,
        )

    def _loss(self, logits, targets, target_lengths):
        # logits:  (B, U-1, V)
        # targets: (B, U)，其中第 0 位是 <sos>
        expected = targets[:, 1:]
        batch_size, steps = expected.shape
        per_token_loss = self.criterion(
            logits.reshape(-1, logits.shape[-1]), expected.reshape(-1)
        ).reshape(batch_size, steps)

        valid_mask = torch.arange(steps, device=logits.device).unsqueeze(0) < (
            target_lengths.to(logits.device).unsqueeze(1) - 1
        )
        return (per_token_loss * valid_mask).sum() / batch_size

    def _teacher_forcing_rate(self):
        """Linearly decay teacher forcing after a configurable start epoch."""
        decay_epochs = max(self.params.teacher_forcing_decay_epochs, 1)
        progress = min(
            max(self.current_epoch - self.params.teacher_forcing_decay_start_epoch, 0)
            / decay_epochs,
            1.0,
        )
        return self.params.teacher_forcing + progress * (
            self.params.teacher_forcing_end - self.params.teacher_forcing
        )

    @staticmethod
    def _edit_distance(reference, hypothesis):
        """Return character-level Levenshtein distance without dependencies."""
        if len(reference) < len(hypothesis):
            reference, hypothesis = hypothesis, reference

        previous = list(range(len(hypothesis) + 1))
        for row, reference_char in enumerate(reference, start=1):
            current = [row]
            for column, hypothesis_char in enumerate(hypothesis, start=1):
                current.append(
                    min(
                        current[-1] + 1,
                        previous[column] + 1,
                        previous[column - 1] + (reference_char != hypothesis_char),
                    )
                )
            previous = current
        return previous[-1]

    def save_checkpoint(self, val_loss, val_cer):
        """Save the latest state and retain a separate best-validation checkpoint."""
        model_to_save = (
            self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        )
        is_best = val_cer < self.best_val_cer
        if is_best:
            self.best_val_loss = val_loss
            self.best_val_cer = val_cer
        state = {
            "next_epoch": self.current_epoch + 1,
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "val_loss": val_loss,
            "val_cer": val_cer,
            "best_val_loss": self.best_val_loss,
            "best_val_cer": self.best_val_cer,
        }

        latest_path = self.checkpoint_dir / "latest.pt"
        torch.save(state, latest_path)

        # Select by the metric used by Kaggle.  NLL remains in the checkpoint
        # for diagnosing optimization, but it is not a proxy for decoding
        # quality under exposure bias.
        if is_best:
            best_path = self.checkpoint_dir / "best.pt"
            torch.save(state, best_path)
            print(
                f"Saved new best checkpoint: {best_path} "
                f"(val_cer={val_cer:.4f}, val_loss={val_loss:.4f})"
            )
        else:
            print(f"Saved latest checkpoint: {latest_path}")

    def load_checkpoint(self, checkpoint_path):
        """Restore a checkpoint created locally by :meth:`save_checkpoint`."""
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        model_to_load = (
            self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        )
        model_to_load.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.current_epoch = int(checkpoint.get("next_epoch", 0))
        self.best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
        self.best_val_cer = float(checkpoint.get("best_val_cer", float("inf")))
        print(
            f"Loaded {checkpoint_path}; "
            f"next epoch={self.current_epoch + 1}, "
            f"best_val_loss={self.best_val_loss:.4f}, "
            f"best_val_cer={self.best_val_cer:.4f}"
        )

    def train(self):
        if self.train_loader is None:
            self._load_train()
        self.model.train()

        total_loss = 0.0
        total_samples = 0
        total_tokens = 0
        teacher_forcing = self._teacher_forcing_rate()

        progress = tqdm(
            self.train_loader,
            desc=f"Train {self.current_epoch + 1}",
            leave=False,
        )

        for features, targets, input_lengths, target_lengths in progress:
            features = features.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.use_amp,
            ):
                logits, _ = self.model(
                    features, input_lengths, targets, teacher_forcing
                )
                loss = self._loss(logits, targets, target_lengths)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            batch_size = features.shape[0]
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            total_tokens += int((target_lengths - 1).sum().item())

            sequence_nll = total_loss / total_samples
            char_nll = total_loss / total_tokens
            progress.set_postfix(
                loss=f"{sequence_nll:.2f}",
                char_nll=f"{char_nll:.3f}",
                ppl=f"{math.exp(min(char_nll, 20.0)):.2f}",
                lr=f"{self.optimizer.param_groups[0]['lr']:.2e}",
            )

        sequence_nll = total_loss / total_samples
        char_nll = total_loss / total_tokens
        print(
            f"Epoch {self.current_epoch + 1}: "
            f"train_loss={sequence_nll:.4f}, "
            f"train_char_nll={char_nll:.4f}, "
            f"train_ppl={math.exp(min(char_nll, 20.0)):.2f}, "
            f"teacher_forcing={teacher_forcing:.2f}"
        )
        return sequence_nll

    # DataParallel creates LSTM replicas lazily.  cuDNN may flatten their
    # weights in-place, which is incompatible with inference_mode tensors.
    @torch.no_grad()
    def validate(self):
        """Evaluate teacher-forced NLL and free-running greedy CER."""
        if self.valid_loader is None:
            self._load_valid()
        self.model.eval()

        total_loss = 0.0
        total_samples = 0
        total_tokens = 0
        total_edits = 0
        total_reference_chars = 0
        total_eos = 0
        total_generated_chars = 0
        progress = tqdm(
            self.valid_loader,
            desc=f"Valid {self.current_epoch + 1}",
            leave=False,
        )

        for features, targets, input_lengths, target_lengths in progress:
            features = features.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.use_amp,
            ):
                logits, _ = self.model(
                    features,
                    input_lengths,
                    targets=targets,
                    teacher_forcing=1.0,
                )
                loss = self._loss(logits, targets, target_lengths)

            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.use_amp,
            ):
                generated_logits, _ = self.model(
                    features,
                    input_lengths,
                    targets=None,
                    teacher_forcing=0.0,
                )
            predicted_ids = generated_logits.argmax(dim=-1).cpu().tolist()
            reference_ids = targets.cpu().tolist()
            eos_id = self.vocab[EOS_TOKEN]
            for prediction, reference in zip(predicted_ids, reference_ids):
                hypothesis_text = self.vocab.decode(prediction, stop_at_eos=True)
                reference_text = self.vocab.decode(reference, stop_at_eos=True)
                total_edits += self._edit_distance(reference_text, hypothesis_text)
                total_reference_chars += len(reference_text)
                if eos_id in prediction:
                    total_eos += 1
                    total_generated_chars += prediction.index(eos_id)
                else:
                    total_generated_chars += len(prediction)

            batch_size = features.shape[0]
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            total_tokens += int((target_lengths - 1).sum().item())

            char_nll = total_loss / total_tokens
            cer = total_edits / max(total_reference_chars, 1)
            progress.set_postfix(
                loss=f"{total_loss / total_samples:.2f}",
                char_nll=f"{char_nll:.3f}",
                ppl=f"{math.exp(min(char_nll, 20.0)):.2f}",
                cer=f"{cer:.3f}",
            )

        sequence_nll = total_loss / total_samples
        char_nll = total_loss / total_tokens
        cer = total_edits / max(total_reference_chars, 1)
        eos_rate = total_eos / max(total_samples, 1)
        mean_generated_chars = total_generated_chars / max(total_samples, 1)
        print(
            f"Epoch {self.current_epoch + 1}: "
            f"val_loss={sequence_nll:.4f}, "
            f"val_char_nll={char_nll:.4f}, "
            f"val_ppl={math.exp(min(char_nll, 20.0)):.2f}, "
            f"greedy_cer={cer:.4f}, "
            f"eos_rate={eos_rate:.3f}, "
            f"mean_pred_chars={mean_generated_chars:.1f}"
        )
        return sequence_nll, cer

    @torch.no_grad()
    def test(self):
        if self.test_loader is None:
            self._load_test()

        self.model.eval()
        predictions = []

        if self.params.decode_strategy == "beam":
            progress = tqdm(total=len(self.test_loader.dataset), desc="Test")
            test_batches = self.test_loader
        else:
            progress = tqdm(self.test_loader, desc="Test")
            test_batches = progress
        for features, input_lengths in test_batches:
            features = features.to(self.device, non_blocking=True)

            if self.params.decode_strategy == "beam":
                # Beam hypotheses carry distinct LSTM states, so searching is
                # done by the underlying module on the primary GPU rather
                # than through DataParallel's scatter/gather wrapper.
                model_to_decode = (
                    self.model.module
                    if isinstance(self.model, nn.DataParallel)
                    else self.model
                )
                decoded = beam_search_batch(
                    model_to_decode,
                    features,
                    input_lengths,
                    eos_index=self.vocab[EOS_TOKEN],
                    beam_width=self.params.beam_width,
                    length_penalty=self.params.beam_length_penalty,
                    on_item_decoded=lambda: progress.update(1),
                )
                predicted_ids = [token_ids for token_ids, _ in decoded]
            else:
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.float16,
                    enabled=self.use_amp,
                ):
                    logits, _ = self.model(
                        features,
                        input_lengths,
                        targets=None,
                        teacher_forcing=0.0,
                    )
                predicted_ids = logits.argmax(dim=-1).cpu().tolist()
            predictions.extend(
                self.vocab.decode(token_ids, stop_at_eos=True)
                for token_ids in predicted_ids
            )

        if self.params.decode_strategy == "beam":
            progress.close()

        output_path = Path(getattr(self.params, "submission_path", "submission.csv"))
        output_path.parent.mkdir(parents=True, exist_ok=True)

        import csv

        with output_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["id", "label"])

            for sample_id, prediction in enumerate(predictions):
                writer.writerow([sample_id, prediction])

        print(f"Saved {len(predictions)} predictions to {output_path}")
        return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument(
        "--override_learning_rate",
        type=float,
        help="Replace the checkpoint optimizer LR after loading it.",
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--teacher_forcing", type=float, default=0.9)
    parser.add_argument(
        "--teacher_forcing_end",
        type=float,
        default=0.6,
        help="Final teacher-forcing probability after the decay period.",
    )
    parser.add_argument(
        "--teacher_forcing_decay_epochs",
        type=int,
        default=12,
        help="Number of epochs over which teacher forcing is linearly decayed.",
    )
    parser.add_argument(
        "--teacher_forcing_decay_start_epoch",
        type=int,
        default=0,
        help="Epoch at which teacher-forcing decay begins.",
    )
    parser.add_argument(
        "--decode",
        choices=("greedy", "beam"),
        default="greedy",
        help="Decoding method used by --test (beam is slower).",
    )
    parser.add_argument("--beam_width", type=int, default=5)
    parser.add_argument("--beam_length_penalty", type=float, default=0.0)
    parser.add_argument(
        "--max_test_samples",
        type=int,
        help="Decode only the first N test utterances for a smoke test; do not submit that CSV.",
    )
    parser.add_argument("--submission_path", type=Path, default=Path("submission.csv"))
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        default=Path("checkpoints/hw4p2"),
        help="Directory for latest.pt and best.pt from this run.",
    )
    parser.add_argument(
        "--gpus",
        help="Comma-separated CUDA IDs for DataParallel, e.g. 0,1.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Checkpoint to resume from or use for --test.",
    )
    args = parser.parse_args()

    gpu_ids = ()
    if args.gpus:
        gpu_ids = tuple(int(gpu_id.strip()) for gpu_id in args.gpus.split(","))

    vocab = CharVocab.from_json(Path(__file__).parent / "utils" / "vocab.json")
    params = ParamsHW4(
        data_dir=args.data_dir,
        device=args.device,
        vocab_size=len(vocab),
        pad_index=vocab[PAD_TOKEN],
        sos_index=vocab[SOS_TOKEN],
        gpu_ids=gpu_ids,
        decode_strategy=args.decode,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
        teacher_forcing=args.teacher_forcing,
        teacher_forcing_end=args.teacher_forcing_end,
        teacher_forcing_decay_epochs=args.teacher_forcing_decay_epochs,
        teacher_forcing_decay_start_epoch=args.teacher_forcing_decay_start_epoch,
        beam_width=args.beam_width,
        beam_length_penalty=args.beam_length_penalty,
        max_test_samples=args.max_test_samples,
        checkpoint_dir=args.checkpoint_dir,
        submission_path=args.submission_path,
    )
    learner = HW4(params, Model1(params), vocab)
    if args.checkpoint is not None:
        learner.load_checkpoint(args.checkpoint)
    if args.override_learning_rate is not None:
        for group in learner.optimizer.param_groups:
            group["lr"] = args.override_learning_rate
        learner.scheduler._last_lr = [
            group["lr"] for group in learner.optimizer.param_groups
        ]
        print(f"Overrode optimizer learning rate to {args.override_learning_rate:.2e}")
    if args.train:
        for epoch in range(learner.current_epoch, params.epochs):
            learner.current_epoch = epoch
            learner.train()
            val_loss, val_cer = learner.validate()
            learner.scheduler.step(val_loss)
            learner.save_checkpoint(val_loss, val_cer)
    if args.validate:
        learner.validate()
    if args.test:
        learner.test()


if __name__ == "__main__":
    main()
