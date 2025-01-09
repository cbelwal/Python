import math

import torch.nn as nn
import torch.nn.functional as F

# From Vaswani: An attention function can be described as mapping a query and a set of key-value pairs to an output,
# where the query, keys, values, and output are all vectors. The output is computed as a weighted sum
# of the values, where the weight assigned to each value is computed by a compatibility function of the
# query with the corresponding key.

# d_model: embedding dimension (pg. 3 of Vaswani et al)
# d_k: dimension of keys
# n_heads: number of attention heads
# In Vaswani:
# d_model: 512
# n_heads = 8, d_k = d_v = d_model/n_heads = 64
# Current desc:
#   d_k = 16,    
#   d_model=64,
#   n_heads=4,
class MultiHeadAttention(nn.Module):
  def __init__(self, d_k, d_model, n_heads):
    super().__init__()

    # Assume d_v = d_k
    self.d_k = d_k
    self.n_heads = n_heads

    # nn.Linear (input features, output features)
    self.key = nn.Linear(d_model, d_k * n_heads)
    self.query = nn.Linear(d_model, d_k * n_heads)
    self.value = nn.Linear(d_model, d_k * n_heads)

    # final linear layer
    # The output for the final layer is equal to embedding size
    # The nn.linear adds trainable weights
    self.fc = nn.Linear(d_k * n_heads, d_model)

  # N - batch size
  # T - sequence length (number of tokens in a sentence)
  def forward(self, q, k, v, mask=None):
    # pass through the nn layers
    q = self.query(q) # N x T x (hd_k), (1,10,64)
    k = self.key(k)   # N x T x (hd_k)
    v = self.value(v) # N x T x (hd_v)

    N = q.shape[0]
    T = q.shape[1]

    # change the shape to:
    # (N, T, h, d_k) -> (N, h, T, d_k)
    # in order for matrix multiply to work properly
    q = q.view(N, T, self.n_heads, self.d_k).transpose(1, 2)
    k = k.view(N, T, self.n_heads, self.d_k).transpose(1, 2)
    v = v.view(N, T, self.n_heads, self.d_k).transpose(1, 2)

    # compute attention weights
    # (N, h, T, d_k) x (N, h, d_k, T) --> (N, h, T, T)
    # @ is a PyTorch Matrix multiplication operator
    attn_scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)
    if mask is not None:
      attn_scores = attn_scores.masked_fill(
          mask[:, None, None, :] == 0, float('-inf'))
    attn_weights = F.softmax(attn_scores, dim=-1)
    
    # compute attention-weighted values
    # (N, h, T, T) x (N, h, T, d_k) --> (N, h, T, d_k)
    A = attn_weights @ v

    # reshape it back before final linear layer
    A = A.transpose(1, 2) # (N, T, h, d_k)
    A = A.contiguous().view(N, T, self.d_k * self.n_heads) # (N, T, h*d_k)

    # projection
    return self.fc(A)