import torch.nn as nn
import import_ipynb
from PositionalEncoding import PositionalEncoding
from TransformerBlock import TransformerBlock

class Encoder(nn.Module):
  def __init__(self,
               vocab_size,
               max_len,
               d_k,
               d_model,
               n_heads,
               n_layers,
               n_classes,
               dropout_prob):
    super().__init__()

    # vocab_size = 20,000
    # max_len = 512
    # d_k = 16
    # d_model = 64
    self.embedding = nn.Embedding(vocab_size, d_model)
    self.pos_encoding = PositionalEncoding(d_model, max_len, dropout_prob)
    transformer_blocks = [
        TransformerBlock(
            d_k,
            d_model,
            n_heads,
            dropout_prob) for _ in range(n_layers)]
    self.transformer_blocks = nn.Sequential(*transformer_blocks)
    self.ln = nn.LayerNorm(d_model)
    self.fc = nn.Linear(d_model, n_classes)
  
  def forward(self, x, mask=None):
    x = self.embedding(x)
    x = self.pos_encoding(x)
    for block in self.transformer_blocks:
      x = block(x, mask)

    # many-to-one (x has the shape N x T x D)
    x = x[:, 0, :]

    x = self.ln(x)
    x = self.fc(x)
    return x
  
