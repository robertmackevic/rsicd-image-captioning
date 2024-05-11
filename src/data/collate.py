from typing import List, Tuple

import torch
from torch import Tensor
from torch.nn import functional

from src.data.vocab import Vocab


def collate_fn(batch: List[Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
    images, captions_batch, caption_lengths = zip(*batch)
    max_caption_length = max(caption.size(1) for caption in captions_batch)

    padded_captions_batch = torch.stack([
        functional.pad(captions, (0, max_caption_length - captions.size(1)), value=Vocab.PAD_ID)
        for captions in captions_batch
    ], dim=0).to(torch.long)

    return torch.stack(images, dim=0), padded_captions_batch, torch.stack(caption_lengths, dim=0)
