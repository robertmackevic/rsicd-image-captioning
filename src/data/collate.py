from typing import List, Tuple

import torch
from torch import Tensor
from torch.nn.functional import pad
from torch.nn.utils.rnn import pad_sequence

from src.data.vocab import Vocab


def collate_fn(batch: List[Tensor]) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    images, captions, caption_lengths, all_captions = zip(*batch)

    images = torch.stack(images, dim=0)

    captions = pad_sequence(captions, batch_first=True, padding_value=Vocab.PAD_ID).to(torch.long)
    caption_lengths = torch.stack(caption_lengths, dim=0).to(torch.long)

    max_caption_length = max(caps.size(1) for caps in all_captions)
    all_captions = torch.stack([
        pad(caps, (0, max_caption_length - caps.size(1)), value=Vocab.PAD_ID)
        for caps in all_captions
    ], dim=0).to(torch.long)

    return images, captions, caption_lengths, all_captions
