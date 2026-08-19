"""Autoregressive search methods for the HW4P2 LAS speller.

Unlike HW3P1's CTC search, LAS has no blank token or path-collapse rule.
Each beam is a distinct character prefix with its own recurrent state.
"""

from __future__ import annotations

import math

import torch


def beam_search(
    decoder,
    keys: torch.Tensor,
    values: torch.Tensor,
    lengths: torch.Tensor,
    eos_index: int,
    beam_width: int = 5,
    max_steps: int | None = None,
    length_penalty: float = 0.0,
):
    """Return the highest-scoring LAS token sequence for one utterance.

    Args:
        decoder: a :class:`models.Decoder` exposing ``initial_state`` and
            ``decode_step``.
        keys, values, lengths: encoder output for *one* utterance.
        eos_index: token ID that terminates a completed hypothesis.
        beam_width: number of candidate prefixes kept at each decoder step.
        length_penalty: optional non-negative score normalization exponent.

    Returns:
        ``(token_ids, log_probability)``.  ``token_ids`` includes ``<eos>``
        when the selected hypothesis generated it.
    """
    if keys.shape[0] != 1:
        raise ValueError("beam_search expects encoder output for exactly one utterance.")
    if beam_width < 1:
        raise ValueError("beam_width must be at least one.")

    if max_steps is None:
        max_steps = decoder.max_decode_steps
    device = keys.device
    initial_token = torch.full(
        (1,), decoder.sos_index, dtype=torch.long, device=device
    )
    # tokens, summed log P(tokens | acoustic input), decoder state, finished
    beams = [((), 0.0, decoder.initial_state(keys, values), initial_token, False)]

    def rank(hypothesis):
        tokens, log_prob, _, _, _ = hypothesis
        if length_penalty == 0.0:
            return log_prob
        # Do not count the terminal EOS as an output character for normalization.
        length = max(len(tokens) - int(bool(tokens and tokens[-1] == eos_index)), 1)
        return log_prob / (length**length_penalty)

    for _ in range(max_steps):
        candidates = []
        active_beams = []
        for beam in beams:
            tokens, log_prob, state, current_token, finished = beam
            if finished:
                candidates.append(beam)
            else:
                active_beams.append(beam)

        if not active_beams:
            break

        # One decoder call evaluates all active hypotheses.  This is the
        # important difference from a literal nested Python loop: width-5
        # beam search now costs roughly one LSTMCell call per time step, not 5.
        active_count = len(active_beams)
        current_tokens = torch.cat([beam[3] for beam in active_beams], dim=0)
        batched_state = {
            name: torch.cat([beam[2][name] for beam in active_beams], dim=0)
            for name in active_beams[0][2]
        }
        logits, next_state, _ = decoder.decode_step(
            keys.expand(active_count, -1, -1),
            values.expand(active_count, -1, -1),
            lengths.expand(active_count),
            current_tokens,
            batched_state,
        )
        all_log_probs = torch.log_softmax(logits, dim=-1)
        # <pad> and <sos> are not valid generated characters.
        all_log_probs[:, decoder.pad_index] = float("-inf")
        all_log_probs[:, decoder.sos_index] = float("-inf")
        top_log_probs, top_ids = torch.topk(
            all_log_probs, k=min(beam_width, all_log_probs.shape[-1]), dim=-1
        )

        for beam_index, (tokens, log_prob, _, _, _) in enumerate(active_beams):
            state_for_prefix = {
                name: value[beam_index : beam_index + 1]
                for name, value in next_state.items()
            }
            for token_log_prob, token_id in zip(
                top_log_probs[beam_index].tolist(), top_ids[beam_index].tolist()
            ):
                token = int(token_id)
                candidates.append(
                    (
                        tokens + (token,),
                        log_prob + float(token_log_prob),
                        state_for_prefix,
                        torch.tensor([token], dtype=torch.long, device=device),
                        token == eos_index,
                    )
                )

        beams = sorted(candidates, key=rank, reverse=True)[:beam_width]
        if all(hypothesis[-1] for hypothesis in beams):
            break

    best_tokens, best_log_prob, _, _, _ = max(beams, key=rank)
    return list(best_tokens), best_log_prob


@torch.no_grad()
def beam_search_batch(model, features, input_lengths, eos_index, on_item_decoded=None, **kwargs):
    """Decode a padded batch by encoding it once and searching each item."""
    keys, values, encoder_lengths = model.encoder(features, input_lengths)
    decoded = []
    for index in range(features.shape[0]):
        decoded.append(
            beam_search(
                model.decoder,
                keys[index : index + 1],
                values[index : index + 1],
                encoder_lengths[index : index + 1],
                eos_index,
                **kwargs,
            )
        )
        if on_item_decoded is not None:
            on_item_decoded()
    return decoded
