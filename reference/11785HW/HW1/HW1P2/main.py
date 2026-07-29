"""Entry point for HW1P2 experiments.

Implement DatasetHW1, LearningHW1, and MLP before running this file.
"""

import argparse

from learninghw1 import LearningHW1, ParamsHW1
from models import MLP


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--gpu_id", default="0")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--context", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    params = ParamsHW1(
        B=args.batch_size,
        K=args.context,
        lr=args.lr,
        max_epoch=args.epochs,
        data_dir=args.data_dir,
        dropout=args.dropout,
        device="cuda:" + args.gpu_id,
    )
    model = MLP(params.K, params.dropout)
    learner = LearningHW1(params, model)
    learner.train()
    learner.test()


if __name__ == "__main__":
    main()
