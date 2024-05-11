from typing import Tuple

from torch import Tensor
from torch.nn import Module, Linear, ReLU, Softmax


class Attention(Module):
    def __init__(self, encoder_dim: int, decoder_dim: int, attention_dim: int) -> None:
        super(Attention).__init__()

        # linear layer to transform the encoded image
        self.encoder_attn = Linear(encoder_dim, attention_dim)
        # linear layer to transform decoder's output
        self.decoder_attn = Linear(decoder_dim, attention_dim)

        # linear layer to calculate logits
        self.full_attn = Linear(attention_dim, 1)

        self.relu = ReLU()
        self.softmax = Softmax(dim=1)

    def forward(self, encoder_output: Tensor, decoder_output: Tensor) -> Tuple[Tensor, Tensor]:
        """
        :param encoder_output: encoded images (batch_size, num_pixels, encoder_dim)
        :param decoder_output: previous decoder output (batch_size, decoder_dim)
        """
        attn1 = self.encoder_attn(encoder_output)
        # (batch_size, num_pixels, attention_dim)

        attn2 = self.decoder_attn(decoder_output)
        # (batch_size, attention_dim)

        attn = self.full_attn(self.relu(attn1 + attn2.unsqueeze(1))).squeeze(2)
        # (batch_size, num_pixels)

        alpha = self.softmax(attn)
        # (batch_size, num_pixels)

        attention_weighted_encoding = (encoder_output * alpha.unsqueeze(2)).sum(dim=1)
        # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha
