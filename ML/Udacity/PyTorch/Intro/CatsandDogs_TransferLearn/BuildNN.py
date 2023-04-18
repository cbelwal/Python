import numpy as np
import torch
from torch import nn

import matplotlib.pyplot as plt
from torch import optim

#Local libs
import helper as helper
#---- Download dataset
from torchvision import datasets, transforms, models

# Define a transform to normalize the data
dataDir = '~/.pytorch/CatsVsDogs/'
# TODO: Define transforms for the training data and testing data
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

# Load the training and testdata that is already downloaded
trainset = datasets.ImageFolder( dataDir + 'train', transform=train_transforms)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32)

testset = datasets.ImageFolder( dataDir + 'test', transform=test_transforms)
testloader = torch.utils.data.DataLoader(testset, batch_size=32)

images, labels = next(iter(trainloader))
helper.imshow(images[0], normalize=False)
#exit() #-- Need to rebuilt model

#----------- Train Model
## Solution
model = models.densenet121(pretrained=True)

# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False

from collections import OrderedDict
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(1024, 500)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(500, 2)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
model.classifier = classifier

criterion = nn.NLLLoss()

optimizer = optim.Adam(model.parameters(), lr=0.003) #Adam is form of SGD with optimizations

# Calculate the loss with the logits and the labels
epochs = 1
steps = 0
running_loss = 0
print_every = 5
for epoch in range(epochs):
    for inputs, labels in trainloader:
        steps += 1
        
        #print("Shape:", inputs.shape)
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(device), labels.to(device)
        
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        
        optimizer.zero_grad() #Important since grads are stored for RNN
        loss.backward()
        optimizer.step() #Updated weights

        running_loss += loss.item()
        
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    
                    test_loss += batch_loss.item()
                    
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Test loss: {test_loss/len(testloader):.3f}.. "
                  f"Test accuracy: {accuracy/len(testloader):.3f}")
            running_loss = 0
            model.train()
