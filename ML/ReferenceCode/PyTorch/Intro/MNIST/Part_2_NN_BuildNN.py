import numpy as np
import torch
from Part_3_NN_Class import Network
from torch import nn

import helper as helper
import matplotlib.pyplot as plt
from torch import optim

#---- Download dataset
from torchvision import datasets, transforms

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

# Load the training data that is already downloaded
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=False, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

#-- Details on images
dataiter = iter(trainloader)
images, labels = next(dataiter)
#images.resize_(images.shape[0], 1, 784)

# Resize images into a 1D vector, new shape is (batch size, color channels, image pixels) 
#images = images.resize_(64, 1, 784)
#images = images.resize_(images.shape[0], 1, 784)
images = images.view(images.shape[0], -1) #Only this works, resize does not



'''
model = nn.Sequential(nn.Linear(784, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 10),
                      nn.LogSoftmax(dim=1))
'''

#----------- Train Model
## Solution
model = Network()
model.initParams()
criterion = nn.NLLLoss()
logps = model(images)
model.initParams()
params = model.parameters()
print(params)
optimizer = optim.SGD(model.parameters(), lr=0.003)
# Calculate the loss with the logits and the labels
loss = criterion(logps, labels)

# exit()
'''
-------------------- Store in temp vars
#limages=[]
#llagles = []
#for images, labels in trainloader:
        # Flatten MNIST images into a 784 long vector
        limages.append(images)
        images = images.view(images.shape[0], -1)
'''

epochs = 5
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        # Flatten MNIST images into a 784 long vector
        images = images.view(images.shape[0], -1)
    
        # TODO: Training pass
        optimizer.zero_grad()
        
        #print('Initial weights - ', model[0].weight)

        #Forward pass, then backward pass
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()

        #Update the weights
        optimizer.step()
        #print('Gradient -', model[0].weight.grad)
        
        running_loss += loss.item()
    else:
        print(f"Training loss: {running_loss/len(trainloader)}")

#Output of proabilities
    #%matplotlib inline
    #import MNIST.helper as helper

    images, labels = next(iter(trainloader))

    img = images[0].view(1, 784)
    # Turn off gradients to speed up this part
    with torch.no_grad():
        logps = model(img)

    # Output of the network are log-probabilities, need to take exponential for probabilities
    ps = torch.exp(logps)
    helper.view_classify(img.view(1, 28, 28), ps)
