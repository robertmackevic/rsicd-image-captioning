from argparse import Namespace
from typing import Tuple

from torch import Tensor, LongTensor
from torch.utils.data import Dataset
from torchrs.datasets import RSICD
from torchvision.transforms import Compose, ToTensor, Resize

from src.data.tokenizer import Tokenizer
from src.data.vocab import Vocab
from src.paths import DATASET_DIR


class RSICDDataset(Dataset):
    def __init__(self, config: Namespace, tokenizer: Tokenizer, split: str) -> None:
        super(Dataset, self).__init__()
        self.data = RSICD(DATASET_DIR, split, Compose([Resize(config.image_size), ToTensor()]))
        self.tokenizer = tokenizer
        self.num_captions_per_image = len(self.data[0]["captions"])

    def __len__(self) -> int:
        return len(self.data) * self.num_captions_per_image

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        image_idx = index // self.num_captions_per_image
        image = self.data[image_idx]["x"]

        all_encoded_captions = [self.tokenizer.encode(cap) for cap in self.data[image_idx]["captions"]]
        max_caption_length = max(len(cap) for cap in all_encoded_captions)

        caption_idx = index % self.num_captions_per_image
        caption = LongTensor(all_encoded_captions[caption_idx])
        caption_length = LongTensor([caption.size(0)])

        all_captions = LongTensor([
            cap + [Vocab.PAD_ID] * (max_caption_length - len(cap))
            for cap in all_encoded_captions
        ])

        return image, caption, caption_length, all_captions
