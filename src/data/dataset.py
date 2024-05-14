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
        caption_idx = index % self.num_captions_per_image
        caption = all_captions[caption_idx]

        if Vocab.UNK_ID in caption:
            # We assume that there's at least one caption that does not contain UNK tokens for an image.
            # This is true for the RSCID dataset.
            valid_captions = [cap for cap in all_captions if Vocab.UNK_ID not in cap]
            caption = valid_captions[0]
            all_captions = valid_captions + [caption] * (self.num_captions_per_image - len(valid_captions))

        caption = LongTensor(caption)
        caption_length = LongTensor([caption.size(0)])

        max_caption_length = max(len(cap) for cap in all_captions)
        all_captions = LongTensor([
            cap + [Vocab.PAD_ID] * (max_caption_length - len(cap))
            for cap in all_captions
        ])

        return image, caption, caption_length, all_captions
