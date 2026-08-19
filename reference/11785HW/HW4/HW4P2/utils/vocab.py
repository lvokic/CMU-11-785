"""Character vocabulary utilities for HW4P2.

Implement a character-level vocabulary from the provided train/dev transcripts.
Do not use test transcripts: they have no labels.
"""

import json
from pathlib import Path
from collections.abc import Iterable, Sequence

PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
SPECIAL_TOKENS = (PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN)


class CharVocab:
    """Bidirectional mapping between transcript characters and integer IDs."""

    def __init__(self, special_tokens: Sequence[str] = SPECIAL_TOKENS):
        self.special_tokens = tuple(special_tokens)
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: list[str] = []

    @staticmethod
    def transcript_to_text(transcript: Sequence[bytes]) -> str:
        """TODO: convert one raw transcript (a sequence of byte words) to text."""
        return " ".join(
            word.decode("utf-8") if isinstance(word, bytes) else str(word)
            for word in transcript
        )

    def build(self, transcripts: Iterable[Sequence[bytes]]) -> None:
        """TODO: collect characters, sort them, and build both mappings."""
        characters = set()
        for transcript in transcripts:
            text = self.transcript_to_text(transcript)
            characters.update(text)
        self.id_to_token = list(self.special_tokens) + sorted(characters)
        self.token_to_id = {
            token: index for index, token in enumerate(self.id_to_token)
        }

    def encode(
        self, text: str, add_sos: bool = True, add_eos: bool = True
    ) -> list[int]:
        """TODO: map text to IDs, using UNK_TOKEN for unseen characters."""
        token_ids = []
        if add_sos:
            token_ids.append(self.token_to_id[SOS_TOKEN])
        uk_id = self.token_to_id[UNK_TOKEN]
        token_ids.extend(self.token_to_id.get(char, uk_id) for char in text)
        if add_eos:
            token_ids.append(self.token_to_id[EOS_TOKEN])
        return token_ids

    def decode(self, token_ids: Sequence[int], stop_at_eos: bool = True) -> str:
        """TODO: map IDs back to text, skipping special tokens as appropriate."""
        characters = []
        for token_id in token_ids:
            token = (
                self.id_to_token[token_id]
                if 0 <= token_id < len(self.id_to_token)
                else UNK_TOKEN
            )
            if token == EOS_TOKEN and stop_at_eos:
                break
            if token in (PAD_TOKEN, SOS_TOKEN, EOS_TOKEN):
                continue
            characters.append(token)
        return "".join(characters)

    def __len__(self) -> int:
        return len(self.id_to_token)

    def __getitem__(self, token: str) -> int:
        """TODO: return a token ID, falling back to UNK_TOKEN."""
        return self.token_to_id.get(token, self.token_to_id[UNK_TOKEN])

    @classmethod
    def from_json(cls, path: Path) -> "CharVocab":
        with path.open(encoding="utf-8") as file:
            legacy_mapping = json.load(file)
        vocab = cls()
        characters = sorted(
            token for token in legacy_mapping if token not in vocab.special_tokens
        )
        vocab.id_to_token = list(vocab.special_tokens) + characters
        vocab.token_to_id = {
            token: index for index, token in enumerate(vocab.id_to_token)
        }
        return vocab
