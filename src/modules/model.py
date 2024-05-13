from argparse import Namespace
from typing import Tuple

from torch import Tensor
from torch.nn import Module

from src.modules.decoder import Decoder
from src.modules.encoder import Encoder


class Image2Text(Module):
    def __init__(self, config: Namespace, vocab_size: int) -> None:
        super(Image2Text, self).__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config, vocab_size, self.encoder.encoder_dim)

    def forward(
            self,
            images: Tensor,
            encoded_captions: Tensor,
            caption_lengths: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        encoder_output = self.encoder(images)
        return self.decoder(encoder_output, encoded_captions, caption_lengths)
