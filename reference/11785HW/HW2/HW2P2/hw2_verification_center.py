"""Starter entry point for the HW2P2 face-verification task."""

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from models import CNN
from hw2_classification import ParamsHW2Classification
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

data_root = Path("./data/hw2p2_data")


class VerificationDataset(Dataset):
    def __init__(self, pair_file, image_root, transform=None):
        super().__init__()

        self.pair_file = Path(pair_file)
        self.image_root = Path(image_root)
        self.transform = transform or transforms.ToTensor()
        self.samples = []

        with self.pair_file.open("r") as file:
            for line in file:
                fields = line.split()
                image1 = fields[0]
                image2 = fields[1]
                label = None
                if len(fields) == 3:
                    label = int(fields[2])
                self.samples.append((image1, image2, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image1_name, image2_name, label = self.samples[index]
        image1_path = self.image_root / image1_name
        image2_path = self.image_root / image2_name

        with Image.open(image1_path) as image:
            image1 = image.convert("RGB")
        with Image.open(image2_path) as image:
            image2 = image.convert("RGB")

        image1 = self.transform(image1)
        image2 = self.transform(image2)

        if label is None:
            return image1, image2
        return image1, image2, torch.tensor(label, dtype=torch.long)


def main():
    val_dataset = VerificationDataset(
        data_root / "val_pairs.txt",
        data_root / "ver_data",
        transform=transforms.ToTensor(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    params = ParamsHW2Classification(
        batch=256,
        lr=1e-3,
        epochs=20,
        data_dir="./data",
        dropout=0.2,
        device=str(device),
    )
    model = CNN(params=params).to(device=device)
    checkpoint_path = Path(
        "./checkpoints/CNN_batch=256_lr=0.001_dropout=0.2e=20.tar"
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("Loaded epoch:", checkpoint["epoch"])
    print("Model device:", device)

    all_scores = []
    all_labels = []
    with torch.no_grad():
        for image1, image2, lables in val_loader:
            image1 = image1.to(device)
            image2 = image2.to(device)

            z1 = model.encode(image1)
            z2 = model.encode(image2)

            z1 = F.normalize(z1, dim=1)
            z2 = F.normalize(z2, dim=1)

            score = torch.sum(z1 * z2, dim=1)
            all_scores.append(score.cpu())
            all_labels.append(lables)

    scores = torch.cat(all_scores).numpy()
    labels = torch.cat(all_labels).numpy()
    auc = roc_auc_score(labels, scores)
    print("Validation AUC:", auc)

    test_dataset = VerificationDataset(
        data_root / "test_pairs.txt",
        data_root / "ver_data",
        transform=transforms.ToTensor(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=4,
        pin_memory=device.type == "cuda",
    )

    test_scores = []
    with torch.inference_mode():
        for image1, image2 in test_loader:
            image1 = image1.to(device, non_blocking=True)
            image2 = image2.to(device, non_blocking=True)

            z1 = F.normalize(model.encode(image1), dim=1)
            z2 = F.normalize(model.encode(image2), dim=1)
            pair_scores = torch.sum(z1 * z2, dim=1)
            test_scores.append(pair_scores.cpu())

    test_scores = torch.cat(test_scores).numpy()
    ids = np.arange(len(test_scores))
    submission = np.column_stack((ids, test_scores))

    np.savetxt(
        "verification_submission.csv",
        submission,
        delimiter=",",
        header="id,score",
        comments="",
        fmt=["%d", "%.8f"],
    )
    print("Saved verification_submission.csv with", len(test_scores), "scores")



if __name__ == "__main__":
    main()
