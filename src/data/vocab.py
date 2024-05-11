from collections import defaultdict

from torchrs.datasets import RSICD


class Vocab:
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    SOS_TOKEN = "<SOS>"
    EOS_TOKEN = "<EOS>"

    def __init__(self, data: RSICD) -> None:
        self.token_to_id = {}
        self.id_to_token = {}

        self._add_token(self.PAD_TOKEN)
        self._add_token(self.UNK_TOKEN)
        self._add_token(self.SOS_TOKEN)
        self._add_token(self.EOS_TOKEN)

        frequencies = defaultdict(int)

        for sample in data:
            for caption in sample["captions"]:
                for token in caption.split():
                    self._add_token(token)
                    frequencies[token] += 1

        self.frequencies = sorted(frequencies.items(), key=lambda item: item[1], reverse=True)

    def __len__(self) -> int:
        return len(self.token_to_id)

    def _add_token(self, token: str) -> None:
        if token not in self.token_to_id:
            token_idx = len(self)
            self.token_to_id[token] = token_idx
            self.id_to_token[token_idx] = token
