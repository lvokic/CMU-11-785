import argparse
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import CNN
from utils.base import Learning, Params


NUM_WORKERS = 4


class ParamsHW2Classification(Params):
    def __init__(self, batch=64, lr=1e-3, epochs=20, data_dir="./data",
                 dropout=0.0, device="cuda:0"):
        super().__init__(B=batch, lr=lr, max_epoch=epochs, data_dir=data_dir,
                         dropout=dropout, device=torch.device(device))

    def __str__(self):
        return f"batch={self.B}_lr={self.lr}_dropout={self.dropout}"


class HW2Classification(Learning):
    def __init__(self, params, model):
        super().__init__(params, model, torch.optim.Adam, nn.CrossEntropyLoss)

    def _load_train(self):
        dataset = datasets.ImageFolder(
            os.path.join(self.params.data_dir, "train_data"),
            transform=transforms.ToTensor(),
        )
        self.train_loader = DataLoader(
            dataset, batch_size=self.params.B, shuffle=True,
            num_workers=NUM_WORKERS, pin_memory=True,
        )

    def _load_valid(self):
        dataset = datasets.ImageFolder(
            os.path.join(self.params.data_dir, "val_data"),
            transform=transforms.ToTensor(),
        )
        self.valid_loader = DataLoader(
            dataset, batch_size=self.params.B, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=True,
        )

    def _load_test(self):
        dataset = datasets.ImageFolder(
            os.path.join(self.params.data_dir, "test_data"),
            transform=transforms.ToTensor(),
        )
        self.test_loader = DataLoader(
            dataset, batch_size=self.params.B, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=True,
        )

    def test(self):
        raise NotImplemented


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--gpu_id", default="0")
    args = parser.parse_args()

    device = "cuda:" + args.gpu_id if torch.cuda.is_available() else "cpu"
    params = ParamsHW2Classification(
        batch=args.batch, lr=args.lr, epochs=args.epochs,
        data_dir=args.data_dir, device=device,
    )
    model = CNN(params)
    learner = HW2Classification(params, model)
    learner.train()


if __name__ == "__main__":
    main()
