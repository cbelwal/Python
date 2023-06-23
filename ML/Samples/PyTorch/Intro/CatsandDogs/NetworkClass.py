import torch.nn.functional as F
from torch import nn
import torch

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        # Defining the layers, 128, 64, 10 units each
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        # Output layer, 10 units - one for each digit
        self.fc4 = nn.Linear(64, 10)

        #Dropout mode
        self.dropout = nn.Dropout(p=0.2)
    
    def initParams(self):
        self.fc1.bias.data.fill_(0)
        self.fc1.weight.data.normal_(std=1)
        
    def forward(self, x):
        ''' Forward pass through the network, returns the output logits '''
        x = x.view(x.shape[0],-1) #Flatten input tensor
        
        # With Dropout 
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        
        # No dropout in output layer
        x = F.log_softmax(self.fc4(x), dim=1)
        
        return x
    
    #prints the state dict for the model
    def printStateDict(self):
        print("Our model: \n\n", self, '\n')
        print("The state dict keys: \n\n", self.state_dict().keys())
    
    def saveModel(self,fileName):
        torch.save(self.state_dict(), fileName)
        print("Model saved to file:",fileName)
        ''' Can save the model directly
        checkpoint = {'input_size': 784,
              'output_size': 10,
              'hidden_layers': [each.out_features for each in model.hidden_layers],
              'state_dict': model.state_dict()}
        '''


    def loadModel(self,fileName):
        state_dict = torch.load(fileName)
        self.load_state_dict(state_dict)
        print("Model loaded from file",fileName)
        '''
         model = fc_model.Network(checkpoint['input_size'],
                             checkpoint['output_size'],
                             checkpoint['hidden_layers'])
          model.load_state_dict(checkpoint['state_dict'])
        '''


