from argparse import Namespace
from typing import Tuple, Optional

from torch import Tensor
from torch.nn import Module

from src.modules.decoder.lstm import LSTMDecoder
from src.modules.decoder.transformer import TransformerDecoder
from src.modules.encoder import Encoder


class Image2Text(Module):
    def __init__(self, config: Namespace, vocab_size: int) -> None:
        super(Image2Text, self).__init__()
        self.encoder = Encoder(config)
        self.decoder_type = config.decoder

        decoder = LSTMDecoder if self.decoder_type == "lstm" else TransformerDecoder
        self.decoder = decoder(config, vocab_size, self.encoder.encoder_dim)

    def forward(
            self,
            images: Tensor,
            captions: Tensor,
            caption_lengths: Optional[Tensor] = None
    ) -> Tuple:
        encoder_output = self.encoder(images)

        if self.decoder_type == "lstm":
            return self.decoder(encoder_output, captions, caption_lengths)

        if self.decoder_type == "transformer":
            return self.decoder(encoder_output, captions)

        raise ValueError(f"Unknown decoder type: `{self.decoder_type}`")
