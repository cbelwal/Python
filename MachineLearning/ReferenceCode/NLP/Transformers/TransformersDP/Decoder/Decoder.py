import torch.nn as nn
import torch.nn.functional as F

from PositionalEncoding import PositionalEncoding
from TransformerBlock import TransformerBlock


# N - batch size 
# T - sequence length (number of tokens in a sentence)
# V - vocab size
class Decoder(nn.Module):
  def __init__(self,
               vocab_size,
               max_len,
               d_k,
               d_model,
               n_heads,
               n_layers,
               dropout_prob):
    super().__init__()

    self.embedding = nn.Embedding(vocab_size, d_model)
    self.pos_encoding = PositionalEncoding(d_model, max_len, dropout_prob)
    transformer_blocks = [
        TransformerBlock(
            d_k,
            d_model,
            n_heads,
            max_len,
            dropout_prob) for _ in range(n_layers)] # n_layers = 6 in the paper
    self.transformer_blocks = nn.Sequential(*transformer_blocks)
    self.ln = nn.LayerNorm(d_model)
    # Output will be of N x T x V
    # or T x V if batch size is 1
    self.fc = nn.Linear(d_model, vocab_size)

  
  def forward(self, x, pad_mask=None):
    x = self.embedding(x)
    x = self.pos_encoding(x)
    for block in self.transformer_blocks:
      x = block(x, pad_mask) # Transfomer block inputs are passed forward to each block
    x = self.ln(x)
    x = self.fc(x) # many-to-many
    return x
  
  def getParamCount(self):
     count = sum(p.numel() for p in self.parameters() if p.requires_grad)
     # The above param count includes all the child layers
     #count += self.transformer_blocks[0].getParamCount() * len(self.transformer_blocks)
     return count