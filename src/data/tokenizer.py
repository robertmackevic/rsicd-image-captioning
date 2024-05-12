import json
from pathlib import Path
from typing import List

from torchrs.datasets import RSICD

from src.data.vocab import Vocab
from src.paths import TOKENIZER_FILE


class Tokenizer:
    def __init__(self, vocab: Vocab) -> None:
        self.vocab = vocab

    @classmethod
    def init_from_data(cls, data: RSICD) -> "Tokenizer":
        return cls(Vocab.init_from_data(data))

    @classmethod
    def init_from_file(cls, filepath: Path = TOKENIZER_FILE) -> "Tokenizer":
        with open(filepath, "r") as file:
            return cls(Vocab(json.load(file)))

    def encode(self, text: str) -> List[int]:
        encoded = [self.vocab.token_to_id.get(token, Vocab.UNK_ID) for token in text.split()]
        return [Vocab.SOS_ID] + encoded + [Vocab.EOS_ID]

    def decode(self, ids: List[int]) -> str:
        return " ".join([self.vocab.id_to_token.get(_id) for _id in ids if _id not in Vocab.SPECIAL_IDS])

    def save(self, filepath: Path) -> None:
        with open(filepath, "w") as file:
            json.dump(self.vocab.token_to_id, file, indent=4)
