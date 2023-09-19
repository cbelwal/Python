# Import libraries
import torch
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

# PyTorch dataset
from torchvision import datasets
import torchvision.transforms as transforms



train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(), # randomly flip and rotate
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


print("Starting data download ...")

#----------- Disable SSL else will get error
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
#---

train_data = datasets.CIFAR10('~/.pytorch/cifar10/', train=True,
                              download=True, transform=train_transform,)
test_data = datasets.CIFAR10('~/.pytorch/cifar10/', train=False,
                             download=True, transform=test_transform)

print("Data download complete")