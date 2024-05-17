import math

import torch
from torch import Tensor
from torch.nn import Module, Dropout


class PositionalEncoding(Module):
    def __init__(self, embedding_dim: int, max_length: int, dropout: float) -> None:
        super(PositionalEncoding, self).__init__()
        self.dropout = Dropout(p=dropout)

        position = torch.arange(max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * (-math.log(10000.0) / embedding_dim))

        pe = torch.zeros(max_length, embedding_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(x + self.pe[:, :x.size(1)])
