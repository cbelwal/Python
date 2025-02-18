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
  # Executed by the function: self.ln1(x + self.mha(x, x, x, mask))
  def forward(self, q, k, v, mask=None):
    # Size of q, k, v: N x T x D, for first data: (1,10,64)
    # pass through the nn layers
    q = self.query(q) # N x T x (hd_k), (1,10,64)
    k = self.key(k)   # N x T x (hd_k)
    v = self.value(v) # N x T x (hd_v)

    N = q.shape[0]
    T = q.shape[1]

      # in order for matrix multiply to work properly
    # tensor.view(): Returns a new tensor with the same data as the orig. tensor but of a different shape.
    # e.g.: 
    # a = torch.range(1, 16)
    # To reshape this tensor to make it a 4 x 4 tensor, use:
    # a = a.view(4, 4)
    # Now 'a' will be a 4 x 4 tensor. Note that after the reshape the total number of elements need to remain the same. 
    # Reshaping the tensor to a 3 x 5 tensor would not be appropriate.
    # for the (1,10,64) tensor: .view(N, T, self.n_heads, self.d_k) will change it to:
    #  (1,10,4,16) and then later .transpose() will change it to:
    #  (1,4,10,16)
    q = q.view(N, T, self.n_heads, self.d_k) # non-verbrose for testing purposes
    # change the shape to: (N, T, h, d_k) -> (N, h, T, d_k)
    q = q.transpose(1, 2) # Note: d_v = d_k
    k = k.view(N, T, self.n_heads, self.d_k).transpose(1, 2) # (1,4,10,16)
    v = v.view(N, T, self.n_heads, self.d_k).transpose(1, 2) # (1,4,10,16)

    # compute attention weights
    # N: Batch size
    # (N, h, T, d_k) x (N, h, d_k, T) --> (N, h, T, T)
    # @ is a PyTorch Matrix multiplication operator
    # Transpose (-2,-1) means that the last two dimensions are swapped
    k = k.transpose(-2, -1) # (1,4,10,16) -> (1,4,16,10)
    # This is doing the dot(.) product between query and keys, and normalizing it by the square root of the dimension of the key
    # The . product will give the highest values for the keys that are most similar to the query
    # There are total T tokens. So each token will have a T scores to represention attention
    # Hence the score matrix will be TxT to represent the attention scores, and then since there  
    # are h attention heads, the final score matrix will be h x T x T
    # Division operation is also called *scaled dot product attention*
    # Scaling insures the values are not too extrmes
    # Scaling is also done in weight initialization for all NNs such that the  
    # variance is 1 and the values are not too extreme
    attn_scores = q @ k / math.sqrt(self.d_k) # attn_scores shape = (1,4,10,10)
    if mask is not None:
      attn_scores = attn_scores.masked_fill(
          mask[:, None, None, :] == 0, float('-inf'))
    # Do a softmax on the last dimension to get the attention weights
    # attention weights will be highest for the keys that are most similar to the query
    attn_weights = F.softmax(attn_scores, dim=-1) # (N,H,T,T)
    
    # compute attention-weighted values
    # (N, h, T, T) x (N, h, T, d_k) --> (N, h, T, d_k)
    # New values are computed at this point
    A = attn_weights @ v

    # reshape it back before final linear layer
    A = A.transpose(1, 2) # (N, h, T, d_k) -> (N, T, h, d_k)
    A = A.contiguous().view(N, T, self.d_k * self.n_heads) # (N, T, h*d_k)

    # projection
    return self.fc(A)