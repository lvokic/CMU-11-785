import argparse
import os
from pathlib import Path

import torch
import numpy as np
from torch import nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from models import CNN
from utils.base import Learning, Params

NUM_WORKERS = 4


def _find_classification_root(data_dir):
    """Find the cls_data directory in either supported data-dir layout."""
    data_dir = Path(data_dir)
    candidates = (
        data_dir / "hw2p2_data" / "cls_data",
        data_dir / "cls_data",
    )
    for root in candidates:
        if (root / "train" / "images").is_dir():
            return root
    raise FileNotFoundError(
        "Could not find classification data. Expected "
        "<data_dir>/hw2p2_data/cls_data/{train,dev,test}/images."
    )


def _infer_num_classes(data_dir, default=8631):
    """Infer the classifier size from train labels, with an S26 fallback."""
    try:
        labels_path = _find_classification_root(data_dir) / "train" / "labels.txt"
    except FileNotFoundError:
        return default

    labels = []
    with labels_path.open("r", encoding="utf-8") as file:
        for line in file:
            fields = line.split()
            if len(fields) >= 2:
                labels.append(int(fields[1]))

    return max(labels) + 1 if labels else default


class FaceClassificationDataset(Dataset):
    """Dataset for S26 classification images and ``filename label`` files."""

    def __init__(self, split_root, transform=None):
        self.split_root = Path(split_root)
        self.images_root = self.split_root / "images"
        self.labels_path = self.split_root / "labels.txt"
        self.transform = transform or transforms.ToTensor()
        self.samples = []

        if not self.images_root.is_dir():
            raise FileNotFoundError(f"Missing image directory: {self.images_root}")
        if not self.labels_path.is_file():
            raise FileNotFoundError(f"Missing labels file: {self.labels_path}")

        with self.labels_path.open("r", encoding="utf-8") as file:
            for line in file:
                fields = line.split()
                if len(fields) < 2:
                    continue
                image_name, label = fields[0], int(fields[1])
                image_path = self.images_root / image_name
                if not image_path.is_file():
                    raise FileNotFoundError(
                        f"Missing image referenced by labels: {image_path}"
                    )
                self.samples.append((image_path, label))

        if not self.samples:
            raise RuntimeError(f"No samples found in {self.labels_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        return self.transform(image), label


class ParamsHW2Classification(Params):
    def __init__(
        self,
        batch=64,
        lr=1e-3,
        epochs=20,
        data_dir="./data",
        dropout=0.0,
        device="cuda:0",
    ):
        num_classes = _infer_num_classes(data_dir)
        super().__init__(
            B=batch,
            lr=lr,
            max_epoch=epochs,
            data_dir=data_dir,
            dropout=dropout,
            device=torch.device(device),
            input_dims=(3, 112, 112),
            output_channels=num_classes,
        )

    def __str__(self):
        return f"batch={self.B}_lr={self.lr}_dropout={self.dropout}"


class HW2Classification(Learning):
    def __init__(self, params, model):
        super().__init__(params, model, torch.optim.Adam, nn.CrossEntropyLoss)

    def _split_dataset(self, split):
        classification_root = _find_classification_root(self.params.data_dir)
        return FaceClassificationDataset(
            classification_root / split,
            transform=transforms.ToTensor(),
        )

    def _load_train(self):
        dataset = self._split_dataset("train")
        self.train_loader = DataLoader(
            dataset,
            batch_size=self.params.B,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

    def _load_valid(self):
        dataset = self._split_dataset("dev")
        self.valid_loader = DataLoader(
            dataset,
            batch_size=self.params.B,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

    def _load_test(self):
        dataset = self._split_dataset("test")
        self.test_loader = DataLoader(
            dataset,
            batch_size=self.params.B,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

    def test(self):
        if self.test_loader is None:
            self._load_test()
        print("Testing...")
        predictions = []
        with torch.cuda.device(self.device):
            with torch.no_grad():
                self.model.eval()
                for i, batch in enumerate(self.test_loader):
                    features = batch[0].to(self.device)
                    logits = self.model(features)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--gpu_id", default="0")
    parser.add_argument(
        "--resume_epoch",
        type=int,
        default=None,
        help="Resume from checkpoints/<model>e=<epoch>.tar",
    )
    args = parser.parse_args()

    device = "cuda:" + args.gpu_id if torch.cuda.is_available() else "cpu"
    params = ParamsHW2Classification(
        batch=args.batch,
        lr=args.lr,
        epochs=args.epochs,
        data_dir=args.data_dir,
        dropout=args.dropout,
        device=device,
    )
    model = CNN(params)
    learner = HW2Classification(params, model)
    if args.resume_epoch is not None:
        learner.load_model(epoch=args.resume_epoch)
        print(f"Resumed from checkpoint epoch {args.resume_epoch}")
    learner.train()


if __name__ == "__main__":
    main()
