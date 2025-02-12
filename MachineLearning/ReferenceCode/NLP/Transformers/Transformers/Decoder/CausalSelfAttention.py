import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
  def __init__(self, d_k, d_model, n_heads, max_len):
    super().__init__()

    # Assume d_v = d_k
    self.d_k = d_k
    self.n_heads = n_heads

    self.key = nn.Linear(d_model, d_k * n_heads)
    self.query = nn.Linear(d_model, d_k * n_heads)
    self.value = nn.Linear(d_model, d_k * n_heads)

    # final linear layer
    self.fc = nn.Linear(d_k * n_heads, d_model)

    # causal mask
    # make it so that diagonal is 0 too
    # this way we don't have to shift the inputs to make targets
    cm = torch.tril(torch.ones(max_len, max_len))
    self.register_buffer(
        "causal_mask",
        cm.view(1, 1, max_len, max_len)
    )

  def forward(self, q, k, v, pad_mask=None):
    q = self.query(q) # N x T x (hd_k)
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
    attn_scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)
    if pad_mask is not None:
      attn_scores = attn_scores.masked_fill(
          pad_mask[:, None, None, :] == 0, float('-inf'))
    attn_scores = attn_scores.masked_fill(
        self.causal_mask[:, :, :T, :T] == 0, float('-inf'))
    attn_weights = F.softmax(attn_scores, dim=-1)
    
    # compute attention-weighted values
    # (N, h, T, T) x (N, h, T, d_k) --> (N, h, T, d_k)
    A = attn_weights @ v

    # reshape it back before final linear layer
    A = A.transpose(1, 2) # (N, T, h, d_k)
    A = A.contiguous().view(N, T, self.d_k * self.n_heads) # (N, T, h*d_k)

    # projection
    return self.fc(A)