import torch

print("Torch version",torch.__version__)
print('Is CUDA available:',torch.cuda.is_available())
print("CUDA Device:", torch.cuda.current_device())