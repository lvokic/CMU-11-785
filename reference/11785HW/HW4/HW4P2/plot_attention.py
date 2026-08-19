"""Plot teacher-forced vanilla LAS attention for a selected dev utterance."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset

from hw4 import DatasetHW4, ParamsHW4, collate_train_val
from models import Model1
from utils.vocab import CharVocab, EOS_TOKEN, PAD_TOKEN, SOS_TOKEN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", type=Path, default=Path("attention.png"))
    args = parser.parse_args()

    device = torch.device(args.device)
    vocab = CharVocab.from_json(Path(__file__).parent / "utils" / "vocab.json")
    params = ParamsHW4(
        data_dir=args.data_dir,
        device=str(device),
        vocab_size=len(vocab),
        pad_index=vocab[PAD_TOKEN],
        sos_index=vocab[SOS_TOKEN],
        amp=False,
    )
    model = Model1(params).to(device).eval()
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    dataset = DatasetHW4(
        args.data_dir / "dev.npy",
        args.data_dir / "dev_transcripts.npy",
        vocab,
    )
    if not 0 <= args.index < len(dataset):
        raise ValueError(f"--index must be in [0, {len(dataset) - 1}]")
    loader = DataLoader(
        Subset(dataset, [args.index]),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_train_val,
    )

    with torch.no_grad():
        features, targets, input_lengths, target_lengths = next(iter(loader))
        logits, attention = model(
            features.to(device),
            input_lengths,
            targets.to(device),
            teacher_forcing=1.0,
        )

    output_steps = int(target_lengths[0].item() - 1)
    encoder_steps = int(input_lengths[0].item() // 8)
    alignment = attention[0, :output_steps, :encoder_steps].cpu()
    token_ids = targets[0, 1 : output_steps + 1].tolist()

    labels = []
    for token_id in token_ids:
        token = vocab.id_to_token[token_id]
        if token == " ":
            token = "␠"
        elif token == EOS_TOKEN:
            token = "<eos>"
        labels.append(token)

    reference = vocab.decode(targets[0].tolist(), stop_at_eos=True)
    teacher_forced_prediction = vocab.decode(
        logits[0].argmax(dim=-1).cpu().tolist(), stop_at_eos=True
    )
    print(f"reference={reference}")
    print(f"teacher_forced_prediction={teacher_forced_prediction}")
    print(f"alignment_shape={tuple(alignment.shape)}")
    print(f"saved={args.output}")

    figure_width = max(12, min(28, output_steps * 0.16))
    figure, axis = plt.subplots(figsize=(figure_width, 7))
    image = axis.imshow(
        alignment.transpose(0, 1),
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap="magma",
    )
    peaks = alignment.argmax(dim=1).numpy()
    axis.plot(range(output_steps), peaks, color="cyan", linewidth=1.0)
    axis.set_xlabel("Transcript character (␠ = space)")
    axis.set_ylabel("Encoder acoustic frame after 3 pBLSTMs")
    axis.set_title(f"Vanilla LAS attention: dev index {args.index}")

    tick_step = max(1, output_steps // 40)
    ticks = list(range(0, output_steps, tick_step))
    axis.set_xticks(ticks)
    axis.set_xticklabels([labels[index] for index in ticks], rotation=90)
    figure.colorbar(image, ax=axis, label="attention weight")
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
