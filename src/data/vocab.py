from typing import Dict

from torchrs.datasets import RSICD


class Vocab:
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    SOS_TOKEN = "<SOS>"
    EOS_TOKEN = "<EOS>"
    PAD_ID = 0
    UNK_ID = 1
    SOS_ID = 2
    EOS_ID = 3
    SPECIAL_IDS = [PAD_ID, UNK_ID, SOS_ID, EOS_ID]

    def __init__(self, token_to_id: Dict[str, int]) -> None:
        self.token_to_id = token_to_id
        self.id_to_token = {value: key for key, value in token_to_id.items()}

    @classmethod
    def init_from_data(cls, data: RSICD) -> "Vocab":
        token_to_id = {
            Vocab.PAD_TOKEN: Vocab.PAD_ID,
            Vocab.UNK_TOKEN: Vocab.UNK_ID,
            Vocab.SOS_TOKEN: Vocab.SOS_ID,
            Vocab.EOS_TOKEN: Vocab.EOS_ID,
        }

        for sample in data:
            for caption in sample["captions"]:
                for token in caption.split():

                    if token not in token_to_id:
                        token_id = len(token_to_id)
                        token_to_id[token] = token_id

        return cls(token_to_id)

    def __len__(self) -> int:
        return len(self.token_to_id)
