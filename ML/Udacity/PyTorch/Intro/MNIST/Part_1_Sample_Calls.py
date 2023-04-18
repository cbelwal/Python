import torch

def activation(x):
    return 1/(1 + torch.exp(-x))

torch.manual_seed(7)

features = torch.randn((1,5))
weights = torch.randn((1,5))

bias = torch.randn((1,1))

y = activation(torch.sum(features * weights) + bias)

if torch.cuda.is_available():
    print("CUDA Available")
else:
    print("CUDA not Available")