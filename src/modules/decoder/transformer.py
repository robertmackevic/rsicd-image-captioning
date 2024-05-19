import warnings
from argparse import Namespace
from typing import Tuple

from torch import Tensor
from torch import nn

from src.data.vocab import Vocab
from src.modules.decoder.encoding import PositionalEncoding
from src.utils import get_available_device

warnings.filterwarnings("ignore", category=UserWarning)


class TransformerDecoder(nn.Module):
    def __init__(self, config: Namespace, vocab_size: int, encoder_dim: int) -> None:
        super(TransformerDecoder, self).__init__()
        config = Namespace(**config.transformer)
        self.device = get_available_device()

        self.embedding = nn.Embedding(vocab_size, encoder_dim)
        self.positional_encoding = PositionalEncoding(encoder_dim, config.max_len, config.dropout)

        self.decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=encoder_dim,
                nhead=config.num_heads,
                dropout=config.dropout,
                batch_first=True
            ),
            num_layers=config.num_layers,
        )

        self.fc = nn.Linear(encoder_dim, vocab_size)
        self._init_weights()

    def _init_weights(self) -> None:
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def forward(self, encoder_output: Tensor, captions: Tensor) -> Tuple[Tensor, Tensor]:
        batch_size = encoder_output.size(0)
        spatial_size = encoder_output.size(1)
        feature_dim = encoder_output.size(3)

        encoder_output = encoder_output.view(batch_size, spatial_size * spatial_size, feature_dim)

        embeddings = self.embedding(captions)
        embeddings = self.positional_encoding(embeddings)
        # [batch_size, caption_length, embedding_dim]

        output = self.decoder(
            tgt=embeddings,
            tgt_mask=nn.Transformer.generate_square_subsequent_mask(embeddings.size(1), device=self.device),
            memory=encoder_output,
            tgt_key_padding_mask=(captions == Vocab.PAD_ID).float(),
        )

        return self.fc(output)
