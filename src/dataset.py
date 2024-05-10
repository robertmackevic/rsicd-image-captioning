from typing import Tuple

from torch import Tensor
from torch.utils.data import Dataset
from torchrs.datasets import RSICD
from torchvision.transforms import Compose, ToTensor, Resize

from src.paths import DATASET_DIR
from src.tokenizer import Tokenizer


class RSICDDataset(Dataset):
    def __init__(self, split: str) -> None:
        super().__init__()
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
