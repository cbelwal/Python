import torch.nn as nn

import MultiHeadAttention

"""
NCHW is an acronym describing the order of the axes in a tensor containing image data samples.

N: Number of data samples.
C: Image channels. A red-green-blue (RGB) image will have 3 channels.
H: Image height.
W: Image width.
"""

# In Vaswani et al it is mentioned "The encoder is composed of a stack of 
# N = 6 identical layers. The transformer block represents each of those 
# N=6 layers
class TransformerBlock(nn.Module):
  def __init__(self, d_k, d_model, n_heads, dropout_prob=0.1):
    super().__init__()

    # torch.nn.LayerNorm(normalized_shape, eps=1e-05, elementwise_affine=True, bias=True, device=None, dtype=None)
    # LayerNorm defined here: https://arxiv.org/abs/1607.06450
    # Layer normalization does not have weights and not used for training 
    self.ln1 = nn.LayerNorm(d_model)
    self.ln2 = nn.LayerNorm(d_model)
    self.mha = MultiHeadAttention(d_k, d_model, n_heads)
    self.ann = nn.Sequential(
        nn.Linear(d_model, d_model * 4),
        nn.GELU(),
        nn.Linear(d_model * 4, d_model),
        nn.Dropout(dropout_prob),
    )
    self.dropout = nn.Dropout(p=dropout_prob)
  
  def forward(self, x, mask=None):
    # The following 2 steps are the two Add & Norm ops defined in
    # Fig. 1 of Vaswani et al.
    x = self.ln1(x + self.mha(x, x, x, mask))
    x = self.ln2(x + self.ann(x))
    x = self.dropout(x)
    return x