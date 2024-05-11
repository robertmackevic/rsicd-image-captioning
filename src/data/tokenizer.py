from typing import List

from torchrs.datasets import RSICD

from src.data.vocab import Vocab


class Tokenizer:
    def __init__(self, data: RSICD) -> None:
        self.vocab = Vocab(data)
        self.pad_id = self.vocab.token_to_id[self.vocab.PAD_TOKEN]
        self.unk_id = self.vocab.token_to_id[self.vocab.UNK_TOKEN]
        self.sos_id = self.vocab.token_to_id[self.vocab.SOS_TOKEN]
        self.eos_id = self.vocab.token_to_id[self.vocab.EOS_TOKEN]

    def encode(self, text: str) -> List[int]:
        encoded = [self.vocab.token_to_id.get(token, self.vocab.UNK_TOKEN) for token in text.split()]
        return [self.sos_id] + encoded + [self.eos_id]

    def decode(self, ids: List[int]) -> str:
        return " ".join([self.vocab.id_to_token.get(_id) for _id in ids][1:-1])
