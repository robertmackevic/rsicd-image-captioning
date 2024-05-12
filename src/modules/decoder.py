from argparse import Namespace
from typing import Tuple

import torch
from torch import Tensor
from torch.nn import Module, Embedding, Dropout, LSTMCell, Linear, Sigmoid

from src.modules.attention import Attention
from src.utils import get_available_device


class Decoder(Module):
    def __init__(self, config: Namespace, vocab_size: int) -> None:
        super(Decoder, self).__init__()
        self.device = get_available_device()
        self.encoder_dim = config.encoder_dim
        self.vocab_size = vocab_size

        self.attention = Attention(self.encoder_dim, config.decoder_dim, config.attention_dim)

        self.embedding = Embedding(self.vocab_size, config.embedding_dim)
        self.dropout = Dropout(p=config.dropout)

        self.decode_step = LSTMCell(config.embedding_dim + self.encoder_dim, config.decoder_dim, bias=True)

        # linear layer to find initial hidden state of LSTMCell
        self.init_h = Linear(self.encoder_dim, config.decoder_dim)
        # linear layer to find initial cell state of LSTMCell
        self.init_c = Linear(self.encoder_dim, config.decoder_dim)
        # linear layer to create a sigmoid-activated gate
        self.f_beta = Linear(config.decoder_dim, self.encoder_dim)

        self.sigmoid = Sigmoid()

        # linear layer to find scores over vocabulary
        self.fc = Linear(config.decoder_dim, self.vocab_size)
        self._init_weights()

    def _init_weights(self) -> None:
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, encoder_out: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.
        :param encoder_out: encoded images (batch_size, num_pixels, encoder_dim)
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        hidden_state = self.init_h(mean_encoder_out)
        # (batch_size, decoder_dim)

        cell_state = self.init_c(mean_encoder_out)
        return hidden_state, cell_state

    def forward(
            self,
            encoder_output: Tensor,
            encoded_captions: Tensor,
            caption_lengths: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        :param encoder_output: encoded images (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions (batch_size, max_caption_length)
        :param caption_lengths: caption lengths (batch_size, 1)
        :return: predictions, sorted encoded captions, decode lengths, weights, sort indices
        """

        batch_size = encoder_output.size(0)
        vocab_size = self.vocab_size

        encoder_out = encoder_output.view(batch_size, -1, self.encoder_dim)
        # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # Sort input data by decreasing lengths; why? apparent below
        caption_lengths, sort_idx = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_idx]
        encoded_captions = encoded_captions[sort_idx]

        embeddings = self.embedding(encoded_captions)
        # (batch_size, max_caption_length, embedding_dim)

        # Initialize LSTM state
        hidden_state, cell_state = self.init_hidden_state(encoder_out)
        # (batch_size, decoder_dim)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word prediction scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(self.device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(self.device)

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):
            batch_size_t = sum([length > t for length in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t], hidden_state[:batch_size_t])

            gate = self.sigmoid(self.f_beta(hidden_state[:batch_size_t]))
            # gating scalar, (batch_size_t, encoder_dim)

            attention_weighted_encoding = gate * attention_weighted_encoding
            hidden_state, cell_state = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (hidden_state[:batch_size_t], cell_state[:batch_size_t])
            )
            # (batch_size_t, decoder_dim)

            pred = self.fc(self.dropout(hidden_state))
            # (batch_size_t, vocab_size)

            predictions[:batch_size_t, t, :] = pred
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas, sort_idx
