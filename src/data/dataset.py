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

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor]:
        image = self.data[index]["x"]
        encoded_captions = [self.tokenizer.encode(caption) for caption in self.data[index]["captions"]]
        caption_lengths = [len(caption) for caption in encoded_captions]
        max_length = max(caption_lengths)

        encoded_and_padded_captions = Tensor([
            encoded + [Vocab.PAD_ID] * (max_length - len(encoded))
            for encoded in encoded_captions
        ])
        return image, encoded_and_padded_captions, LongTensor(caption_lengths)
