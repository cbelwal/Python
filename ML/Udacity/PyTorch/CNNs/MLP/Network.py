import torch.nn.functional as F
from torch import nn

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        # Defining the layers, 128, 64, 10 units each
        self.fc1 = nn.Linear(784, 1)
        
    def forward(self, x):
        ''' Forward pass through the network, returns the output logits '''
         # flatten image input
        x = x.view(-1, 28 * 28)
        # add hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        return x
        

    
 


