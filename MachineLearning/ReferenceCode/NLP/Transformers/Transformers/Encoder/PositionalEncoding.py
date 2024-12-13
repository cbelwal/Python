import torch
import torch.nn as nn

import math

# d_model: embedding dimension (pg. 3 of Vaswani et al)
class PositionalEncoding(nn.Module):
  # max_len: max size of positional embeddings
  def __init__(self, d_model, max_len=2048, dropout_prob=0.1):
    super().__init__()
    self.dropout = nn.Dropout(p=dropout_prob)

    # unsqueeze: position dimension will be 1
    position = torch.arange(max_len).unsqueeze(1)
    
    # torch.arange(start=0, end, step=1)
    # Returns a 1-D tensor of size ⌈end−start/step⌉ with values from the 
    # interval [start, end) taken with common difference step beginning from start.
    # eg. if d_model = 6, exp_term = [0,2,4]
    d_model = 6
    max_len = 2
    exp_term = torch.arange(0, d_model, 2)
    
    # div_term gives different frequencies
    print(math.log(100.0))
    # x = -math.log(10000.0) = -9.210340371976184, e^x = 10000, e = 2.71
    
    #torch.exp(x)= e(x)
    div_term = torch.exp(exp_term * (-math.log(10000.0) / d_model))
    # torch.zeros(*size), dimensions of the tensor
    # will allocate a tensor of dimensions max_len, d_model 
    pe = torch.zeros(1, max_len, d_model)

    # Follow the format start:stop:step]
    # with 0::2: Start from 0 till end and assign indexes at step 2: 0,2,4,5
    # with 1::2: Start from 1 till end and assign indexes at step 2: 1,3,5,7
    # This matches up with the size of exp_term
    # position is a scalar with values from 0 
    pe[0, :, 0::2] = torch.sin(position * div_term)
    pe[0, :, 1::2] = torch.cos(position * div_term)
    
    # register_buffer:This is typically used to register a buffer that 
    # should not to be considered a model parameter. For example, 
    # BatchNorm’s running_mean is not a parameter, but is part of the 
    # module’s state. Buffers, by default, are persistent and will be 
    # saved alongside parameters. This behavior can be changed by setting 
    # persistent to False. The only difference between a persistent 
    # buffer and a non-persistent buffer is that the latter will not be a 
    # part of this module’s state_dict.
    self.register_buffer('pe', pe)

  def forward(self, x):
    # x.shape: N x T x D
    x = x + self.pe[:, :x.size(1), :]
    return self.dropout(x)

if __name__ == "__main__":
  pe = PositionalEncoding(1536)
