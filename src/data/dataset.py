from argparse import Namespace
from typing import Tuple

from torch import Tensor, LongTensor
from torch.utils.data import Dataset
from torchrs.datasets import RSICD

from src.data.tokenizer import Tokenizer
from src.data.vocab import Vocab
from src.inference import compose_image_transform
from src.paths import DATASET_DIR


class RSICDDataset(Dataset):
    def __init__(self, config: Namespace, tokenizer: Tokenizer, split: str) -> None:
        super(Dataset, self).__init__()
        self.data = RSICD(DATASET_DIR, split, compose_image_transform(config.image_size))
        self.tokenizer = tokenizer
        self.num_captions_per_image = len(self.data[0]["captions"])

    def __len__(self) -> int:
        return len(self.data) * self.num_captions_per_image

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        image_idx = index // self.num_captions_per_image
        image = self.data[image_idx]["x"]

        all_captions = [self.tokenizer.encode(cap) for cap in self.data[image_idx]["captions"]]
        valid_captions = [
            cap
            for cap in all_captions
            if Vocab.UNK_ID not in cap and len(self.tokenizer.decode(cap)) > 1
        ]

        caption_idx = index % self.num_captions_per_image
        caption = all_captions[caption_idx]

        if Vocab.UNK_ID in caption or len(self.tokenizer.decode(caption)) < 2:
            try:
                caption = valid_captions[0]
            # This is a case to handle "bad" captions that exists primarily in the validation set
            # If not handled ROUGE score calculation will not work, because the captions are empty.
            except IndexError:
                return self.__getitem__(0)

        if len(valid_captions) != self.num_captions_per_image:
            valid_captions += [caption] * (self.num_captions_per_image - len(valid_captions))

        caption = LongTensor(caption)
        caption_length = LongTensor([caption.size(0)])

        max_caption_length = max(len(cap) for cap in valid_captions)
        valid_captions = LongTensor([
            cap + [Vocab.PAD_ID] * (max_caption_length - len(cap))
            for cap in valid_captions
        ])

        return image, caption, caption_length, valid_captions
