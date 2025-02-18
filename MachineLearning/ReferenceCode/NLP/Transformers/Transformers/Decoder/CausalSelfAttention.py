import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# This is also know as masked self-attention
# It is causal as it requires the model to only look at the past
# Causality will imply to adjust the attention weights matrix to not look at the future
# 
# To accomplish this, we need to make sure that the attention weights matrix is lower triangular
# or for a cell A(i,j), which is how much token i should pay attention to token j 
# should have value 0 if i < j (or when i comes before j)
# and value != 0 when i >= j (or when i comes after j)
#  
# This is accomplished by using a causal which is a lower triangular matrix
# as the lower triangular matrix will have 0s in the upper triangle 
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
    # torch.tril: Returns the lower triangular part of the matrix (2-D tensor) with the elements above the diagonal set to 0.
    cm = torch.tril(torch.ones(max_len, max_len))
    # register_buffer: Adds a persistent buffer to the module.
    # register_buffer is PyTorch’s way of letting you store non-trainable 
    # tensors within a model, which are not used in weight updates
    #
    # register_buffer(name, tensor, persistent=True): the name of the variable is not part of the module
    self.register_buffer(
        "causal_mask", # Name of variable
        # .view changes the dimensions of the tensor
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
    
    # Tensor.masked_fill_(mask, value)
    # Fills elements of self tensor with value where mask is True.
    # 'causal_mask' is defined in the constructor in register buffer
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