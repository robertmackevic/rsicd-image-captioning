from typing import Tuple, List

import torch
from torch import Tensor
from torch.nn import functional
from torch.utils.data import Dataset
from torchrs.datasets import RSICD
from torchvision.transforms import Compose, ToTensor, Resize

from src.data.tokenizer import Tokenizer
from src.paths import DATASET_DIR


class RSICDDataset(Dataset):
    def __init__(self, split: str) -> None:
        super(Dataset).__init__()
        self.data = RSICD(DATASET_DIR, split, Compose([Resize((224, 224)), ToTensor()]))
        self.tokenizer = Tokenizer(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        image = self.data[index]["x"]
        encoded_captions = [self.tokenizer.encode(caption) for caption in self.data[index]["captions"]]
        max_tokens = len(max(encoded_captions, key=len))
        encoded_and_padded_captions = Tensor([
            encoded + [self.tokenizer.pad_id] * (max_tokens - len(encoded))
            for encoded in encoded_captions
        ])
        return image, encoded_and_padded_captions

    def collate_fn(self, batch: List[Tensor]) -> Tuple[Tensor, Tensor]:
        images, captions_batch = zip(*batch)
        max_caption_length = max(caption.size(1) for caption in captions_batch)

        padded_captions_batch = torch.stack([
            functional.pad(captions, (0, max_caption_length - captions.size(1)), value=self.tokenizer.pad_id)
            for captions in captions_batch
        ], dim=0)

        return torch.stack(images, dim=0), padded_captions_batch
