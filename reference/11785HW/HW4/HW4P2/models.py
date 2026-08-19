"""HW4P2 LAS model components (clean handout reconstruction)."""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pad_packed_sequence,
)


class LockedDropout(nn.Module):
    """Drop the same feature channels at every timestep."""

    def __init__(self, p=0.0):
        super().__init__()
        self.p = float(p)

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        if not 0.0 <= self.p < 1.0:
            raise ValueError(f"LockedDropout p must be in [0, 1), got {self.p}")
        mask = x.new_empty(x.shape[0], 1, x.shape[2]).bernoulli_(1.0 - self.p)
        mask = mask.div_(1.0 - self.p)
        return x * mask


class PBLSTM(nn.Module):
    """TODO: halve time resolution, then run a bidirectional LSTM."""

    def __init__(self, input_size, hidden_size, dropout=0.0):
        super().__init__()
        self.blstm = nn.LSTM(
            input_size=input_size * 2,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.locked_dropout = LockedDropout(dropout)

    def forward(self, x, lengths):
        """
        x:       (B, T, D)
        lengths: (B,): padding 前每条 utterance 的真实帧数

        returns:
            output:      (B, floor(T / 2), 2 * hidden_size)
            new_lengths: (B,)
        """
        batch_size, time_steps, feature_dim = x.shape
        if time_steps % 2 == 1:
            x = x[:, :-1, :]
            time_steps -= 1
        x = x.reshape(batch_size, time_steps // 2, feature_dim * 2)
        x = self.locked_dropout(x)
        new_lengths = lengths.to(torch.long) // 2
        packed = pack_padded_sequence(
            x, new_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.blstm(packed)
        output, _ = pad_packed_sequence(
            packed_output, batch_first=True, total_length=time_steps // 2
        )
        return output, new_lengths


class Encoder(nn.Module):
    """TODO: listener network yielding attention keys, values, and lengths."""

    def __init__(self, params):
        super().__init__()
        feature_dim = params.feature_dim
        hidden_size = params.encoder_hidden
        attention_dim = params.attention_dim
        dropout = params.dropout

        self.input_dropout = LockedDropout(dropout)
        self.base_blstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.pblstm_layers = nn.ModuleList(
            [
                PBLSTM(
                    input_size=hidden_size * 2,
                    hidden_size=hidden_size,
                    dropout=dropout,
                )
                for _ in range(3)
            ]
        )
        self.key_projection = nn.Linear(hidden_size * 2, attention_dim)
        self.value_projection = nn.Linear(hidden_size * 2, attention_dim)

    def forward(self, x, lengths):
        x = self.input_dropout(x)
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed, _ = self.base_blstm(packed)
        x, _ = pad_packed_sequence(packed, batch_first=True, total_length=x.shape[1])
        for pblstm in self.pblstm_layers:
            x, lengths = pblstm(x, lengths)
        keys = self.key_projection(x)
        values = self.value_projection(x)

        return keys, values, lengths


class Attention(nn.Module):
    """Single-head dot-product content attention used by LAS."""

    def __init__(self, key_size, query_size):
        super().__init__()
        self.query_projection = nn.Linear(query_size, key_size)

    def forward(self, keys, values, query, lengths):
        projected_query = self.query_projection(query)
        scores = torch.bmm(keys, projected_query.unsqueeze(-1)).squeeze(-1)
        time_steps = keys.shape[1]
        valid_mask = torch.arange(time_steps, device=keys.device).unsqueeze(
            0
        ) < lengths.to(keys.device).unsqueeze(1)
        scores = scores.masked_fill(~valid_mask, float("-inf"))
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), values).squeeze(1)

        return context, weights


class Decoder(nn.Module):
    """TODO: combine listener encoder and attention-based speller decoder."""

    def __init__(self, params):
        super().__init__()
        self.vocab_size = params.vocab_size
        self.hidden_size = params.decoder_hidden
        self.attention_dim = params.attention_dim
        self.pad_index = params.pad_index
        self.sos_index = params.sos_index
        self.max_decode_steps = params.max_decode_steps

        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=params.embedding_size,
            padding_idx=params.pad_index,
        )
        # The writeup's speller is a two-layer LSTM.  Only the first layer
        # receives the previous attention context; its output feeds layer 2.
        self.lstm_cell_1 = nn.LSTMCell(
            input_size=params.embedding_size + params.attention_dim,
            hidden_size=self.hidden_size,
        )
        self.lstm_cell_2 = nn.LSTMCell(
            input_size=self.hidden_size,
            hidden_size=params.attention_dim,
        )
        self.second_hidden_size = params.attention_dim
        if params.embedding_size != 2 * params.attention_dim:
            raise ValueError(
                "Weight tying requires embedding_size == 2 * attention_dim; "
                f"got {params.embedding_size} and {params.attention_dim}."
            )
        self.attention = Attention(
            params.attention_dim,
            params.attention_dim,
        )
        self.classifier = nn.Linear(params.embedding_size, self.vocab_size)
        # Share the output projection weight with the token embedding.
        self.classifier.weight = self.embedding.weight

    def initial_state(self, keys, values):
        """Create the per-hypothesis recurrent state used by decoding."""
        batch_size = keys.shape[0]
        state = {
            "hidden_1": keys.new_zeros(batch_size, self.hidden_size),
            "cell_1": keys.new_zeros(batch_size, self.hidden_size),
            "hidden_2": keys.new_zeros(batch_size, self.second_hidden_size),
            "cell_2": keys.new_zeros(batch_size, self.second_hidden_size),
            "context": values.new_zeros(batch_size, self.attention_dim),
        }
        return state

    def decode_step(self, keys, values, lengths, current_token, state):
        """Decode one autoregressive step.

        ``state`` is returned rather than modified in place, so several beam
        candidates may safely branch from the same previous hypothesis.
        """
        embedded = self.embedding(current_token)
        decoder_input = torch.cat((embedded, state["context"]), dim=-1)
        hidden_1, cell_1 = self.lstm_cell_1(
            decoder_input, (state["hidden_1"], state["cell_1"])
        )
        hidden_2, cell_2 = self.lstm_cell_2(
            hidden_1, (state["hidden_2"], state["cell_2"])
        )
        context, weights = self.attention(keys, values, hidden_2, lengths)
        logits = self.classifier(
            torch.cat((hidden_2, context), dim=-1)
        )
        next_state = {
            "hidden_1": hidden_1,
            "cell_1": cell_1,
            "hidden_2": hidden_2,
            "cell_2": cell_2,
            "context": context,
        }
        return logits, next_state, weights

    def forward(self, keys, values, lengths, targets=None, teacher_forcing=0.0):
        """
        keys, values: (B, T_encoder, attention_dim)
        lengths:      (B,)

        targets:      (B, U)，格式为 <sos> text <eos>；训练/验证时提供
        teacher_forcing: 使用真实上一个字符作为下步输入的概率

        returns:
            logits:            (B, steps, vocab_size)
            attention_weights: (B, steps, T_encoder)
        """
        batch_size = keys.shape[0]
        device = keys.device
        if targets is None:
            steps = self.max_decode_steps
            current_token = torch.full(
                (batch_size,),
                self.sos_index,
                dtype=torch.long,
                device=device,
            )
        else:
            steps = targets.shape[1] - 1
            current_token = targets[:, 0]

        state = self.initial_state(keys, values)

        logits_per_step = []
        weights_per_step = []

        for step in range(steps):
            logits, state, weights = self.decode_step(
                keys, values, lengths, current_token, state
            )
            logits_per_step.append(logits)
            weights_per_step.append(weights)
            predicted_token = logits.argmax(dim=-1)

            if targets is None:
                current_token = predicted_token
            elif step + 1 < targets.shape[1]:
                use_teacher = torch.rand(batch_size, device=device) < teacher_forcing
                current_token = torch.where(
                    use_teacher, targets[:, step + 1], predicted_token
                )

        logits = torch.stack(logits_per_step, dim=1)
        attention_weights = torch.stack(weights_per_step, dim=1)

        return logits, attention_weights


class Model1(nn.Module):
    """TODO: character embedding, attention-aware LSTMCell, and vocabulary logits."""

    def __init__(self, params):
        super().__init__()
        self.encoder = Encoder(params)
        self.decoder = Decoder(params)

    def forward(self, x, lengths, targets=None, teacher_forcing=0.0):
        keys, values, encoder_lengths = self.encoder(x, lengths)
        logits, attention_weights = self.decoder(
            keys, values, encoder_lengths, targets, teacher_forcing
        )
        return logits, attention_weights
