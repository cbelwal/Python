import numpy as np
import torch
from torch import nn

import matplotlib.pyplot as plt
from torch import optim

#Local libs
from NetworkClass import Network
import helper as helper
#---- Download dataset
from torchvision import datasets, transforms

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])
# Load the training and testdata that is already downloaded
trainset = datasets.FashionMNIST('~/.pytorch/FashionMNIST_data/', download=False, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = datasets.FashionMNIST('~/.pytorch/FashionMNIST_data/', download=False, train=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)


#----------- Train Model
## Solution
model = Network()
model.initParams()
criterion = nn.NLLLoss(reduction='sum')

optimizer = optim.Adam(model.parameters(), lr=0.003) #Adam is form of SGD with optimizations

#Total Train Sample
print("Train Samples:", len(trainloader.dataset))
print("Test Samples:", len(testloader.dataset))


#exit()
# Calculate the loss with the logits and the labels
epochs = 0
train_losses, test_losses = [], []
for e in range(epochs):
    tot_train_loss = 0
    for images, labels in trainloader:
        # Flatten is inside the NN
        optimizer.zero_grad()
        
        logps = model(images)
        loss = criterion(logps, labels)

        tot_train_loss += loss.item()        
        #Update the weights
        loss.backward()
        optimizer.step()


    else: #This is run after the for loop
        #Validation loss
        tot_test_loss=0
        test_correct = 0
        #print(f"Training loss: {running_loss/len(trainloader)}")
        with torch.no_grad():
            model.eval()
            for images, labels in testloader:
                logps = model(images)
                loss = criterion(logps, labels) #Pass it the logits function
                tot_test_loss += loss.item()

                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1) 
                equals = top_class == labels.view(*top_class.shape) #True Positive Rate
                test_correct += equals.sum().item()
                accuracy = torch.mean(equals.type(torch.FloatTensor))
        
         # Get mean loss to enable comparison between train and test sets
        train_loss = tot_train_loss / len(trainloader.dataset)
        test_loss = tot_test_loss / len(testloader.dataset)

        # At completion of epoch
        train_losses.append(train_loss)
        test_losses.append(test_loss)

        print("Epoch: {}/{}.. ".format(e+1, epochs),
              "Training Loss: {:.3f}.. ".format(train_loss),
              "Test Loss: {:.3f}.. ".format(test_loss),
              "Test Accuracy: {:.3f}".format(test_correct / len(testloader.dataset)))


#------
fileName = "./modelCheckPoint.pth"
print("----- Saving Model")
model.printStateDict()
model.saveModel(fileName)
model.loadModel(fileName)
#-------------------------


# Plot Trains vs Validation loss
import matplotlib.pyplot as plt

plt.plot(train_losses, label='Training loss')
plt.plot(test_losses, label='Validation loss')
plt.legend(frameon=False)
plt.show()

#----------------- Plotting
#Output of proabilities
    #%matplotlib inline
    #import MNIST.helper as helper
'''
images, labels = next(iter(trainloader))
img = images[0].view(1, 784)
# Turn off gradients to speed up this part
with torch.no_grad():
    logps = model(img)
# Output of the network are log-probabilities, need to take exponential for probabilities
ps = torch.exp(logps)
helper.view_classify(img.view(1, 28, 28), ps,version = 'Fashion')
'''