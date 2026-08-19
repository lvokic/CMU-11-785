"""Teacher-forced LAS attention diagnostics (no decoding or CER)."""

import argparse
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from hw4 import DatasetHW4, HW4, ParamsHW4, collate_train_val
from models import Model1
from utils.vocab import CharVocab, PAD_TOKEN, SOS_TOKEN


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_samples", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--print_examples",
        type=int,
        default=0,
        help="Print the first N dev samples; ignored when --example_indices is set.",
    )
    parser.add_argument(
        "--example_indices",
        type=str,
        help="Comma-separated dev indices to print, e.g. 0,7,15,31.",
    )
    args = parser.parse_args()
    if args.example_indices:
        example_indices = {
            int(value.strip())
            for value in args.example_indices.split(",")
            if value.strip()
        }
    else:
        example_indices = set(range(max(args.print_examples, 0)))

    device = torch.device(args.device)
    vocab = CharVocab.from_json(Path(__file__).parent / "utils" / "vocab.json")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    params = ParamsHW4(
        data_dir=args.data_dir,
        device=str(device),
        vocab_size=len(vocab),
        pad_index=vocab[PAD_TOKEN],
        sos_index=vocab[SOS_TOKEN],
    )
    model = Model1(params).to(device).eval()
    model.load_state_dict(checkpoint["model_state_dict"])

    dataset = DatasetHW4(
        args.data_dir / "dev.npy",
        args.data_dir / "dev_transcripts.npy",
        vocab,
    )
    count = min(args.num_samples, len(dataset))
    loader = DataLoader(
        Subset(dataset, range(count)),
        batch_size=min(args.batch_size, count),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_train_val,
    )

    peaks, entropies, backward_rates, spans, end_positions = [], [], [], [], []
    examples = []
    decoded_examples = []
    sample_offset = 0

    def repetition_rate(text):
        if len(text) < 2:
            return 0.0
        bigrams = [text[index : index + 2] for index in range(len(text) - 1)]
        return 1.0 - len(set(bigrams)) / len(bigrams)

    def max_character_run(text):
        if not text:
            return 0
        best = current = 1
        for previous, current_char in zip(text, text[1:]):
            current = current + 1 if previous == current_char else 1
            best = max(best, current)
        return best

    for features, targets, input_lengths, target_lengths in loader:
        features = features.to(device)
        targets = targets.to(device)
        _, weights = model(features, input_lengths, targets=targets, teacher_forcing=1.0)
        generated_logits, _ = model(
            features, input_lengths, targets=None, teacher_forcing=0.0
        )
        predicted_ids = generated_logits.argmax(dim=-1).cpu().tolist()
        reference_ids = targets.cpu().tolist()
        # Three pBLSTM layers reduce each utterance by floor(T / 8).
        encoder_lengths = input_lengths // 8

        for index in range(features.shape[0]):
            # Decoder output steps are ``text characters + <eos>``.  The EOS
            # prediction can reasonably attend anywhere because it is often
            # inferred from the decoded history, so exclude it when judging
            # acoustic alignment.
            output_steps = int(target_lengths[index].item() - 1)
            content_steps = max(output_steps - 1, 1)
            encoder_steps = int(encoder_lengths[index].item())
            attention = weights[index, :content_steps, :encoder_steps].float()
            locations = attention.argmax(dim=1)
            peak = attention.max(dim=1).values.mean().item()
            if encoder_steps > 1:
                entropy = (
                    -(attention.clamp_min(1e-12) * attention.clamp_min(1e-12).log())
                    .sum(dim=1)
                    .mean()
                    .item()
                    / math.log(encoder_steps)
                )
                span = (locations.max() - locations.min() + 1).item() / encoder_steps
                end_position = locations[-1].item() / (encoder_steps - 1)
            else:
                entropy, span, end_position = 0.0, 1.0, 1.0
            if output_steps > 1:
                # A small local revisit is harmless; count meaningful moves
                # backwards by two or more encoder frames.
                backward_rate = (locations[1:] < locations[:-1] - 1).float().mean().item()
            else:
                backward_rate = 0.0

            peaks.append(peak)
            entropies.append(entropy)
            backward_rates.append(backward_rate)
            spans.append(span)
            end_positions.append(end_position)
            if len(examples) < 3:
                examples.append((content_steps, encoder_steps, peak, entropy, backward_rate, span, end_position))

            sample_index = sample_offset + index
            if sample_index in example_indices:
                hypothesis = vocab.decode(predicted_ids[index], stop_at_eos=True)
                reference = vocab.decode(reference_ids[index], stop_at_eos=True)
                decoded_examples.append(
                    {
                        "index": sample_index,
                        "reference": reference,
                        "hypothesis": hypothesis,
                        "cer": HW4._edit_distance(reference, hypothesis)
                        / max(len(reference), 1),
                        "ref_len": len(reference),
                        "hyp_len": len(hypothesis),
                        "repeat_bigram": repetition_rate(hypothesis),
                        "max_char_run": max_character_run(hypothesis),
                    }
                )

        sample_offset += features.shape[0]

    print(f"checkpoint={args.checkpoint}, dev_samples={count}, attention=vanilla")
    print(f"mean_attention_peak={sum(peaks) / len(peaks):.4f}")
    print(f"mean_normalized_entropy={sum(entropies) / len(entropies):.4f}  (1.0 = uniform)")
    print(f"mean_backward_jump_rate={sum(backward_rates) / len(backward_rates):.4f}")
    print(f"mean_attended_span={sum(spans) / len(spans):.4f}  (1.0 = whole utterance)")
    print(f"mean_final_attention_position={sum(end_positions) / len(end_positions):.4f}  (1.0 = final encoder frame)")
    print("\n--- decoded examples ---")
    for item in decoded_examples:
        print(
            f"sample[{item['index']}] cer={item['cer']:.3f} "
            f"ref_len={item['ref_len']} hyp_len={item['hyp_len']} "
            f"repeat_bigram={item['repeat_bigram']:.3f} "
            f"max_char_run={item['max_char_run']}"
        )
        print(f"  REF: {item['reference']}")
        print(f"  HYP: {item['hypothesis']}")
    for index, values in enumerate(examples):
        u, t, peak, entropy, backward, span, end = values
        print(
            f"sample[{index}]: target_steps={u}, encoder_steps={t}, "
            f"peak={peak:.3f}, entropy={entropy:.3f}, backward={backward:.3f}, "
            f"span={span:.3f}, final_position={end:.3f}"
        )


if __name__ == "__main__":
    main()
