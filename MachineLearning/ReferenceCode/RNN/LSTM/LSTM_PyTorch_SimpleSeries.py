# This is the same LSTM as used in LSTM_Scratch_PyTorch_SimpleSeries.py
# except it uses the native LSTM in PyTorch.

import torch # torch will allow us to create tensors.
import torch.nn as nn # torch.nn allows us to create a neural network.
import torch.nn.functional as F # nn.functional give us access to the activation and loss functions.
from torch.optim import Adam # optim contains many optimizers. This time we're using Adam

from torch.utils.data import TensorDataset, DataLoader # these are needed for the training data


#inherit from nn.Module
class CustomLSTM (nn.Module):
    def __init__(self):
        super().__init__() # Base class init  
        # input_size: Number of featrues/variables in input data. Here it is 1
        # hidden_size: Number of output values we want.
        # extra weights and biases are added to accomodate the hidden_size
        self.lstm = nn.LSTM(input_size=1,hidden_size=1 )

    def initParams(self):
        torch.manual_seed(0) # So the same initial weights are used in every run

    # Forward function is also called when model infers
    def forward(self, inputs):
        # Transpose to a column matrix (currently all values are in single row)
        input_trans = inputs.view(len(inputs),1) # rows<-len(inputs), cols<-1
        lstm_out, temp = self.lstm(input_trans)

        # lstm_out is an array
        # lstm_out will contain all values from each step when LSTM is unrolled
        prediction = lstm_out[-1] # last value
        return prediction

    def getOptimizer(self): # this configures the optimizer we want to use for backpropagation.
        # return Adam(self.parameters(), lr=0.1) # NOTE: Setting the learning rate to 0.1 trains way faster than
                                                 # using the default learning rate, lr=0.001, which requires a lot more 
                                                 # training. However, if we use the default value, we get 
                                                 # the exact same Weights and Biases that I used in
                                                 # the LSTM Clearly Explained StatQuest video. So we'll use the
                                                 # default value.
        return Adam(self.parameters())
    
    def training_step(self, batch, batch_idx=0): # take a step during gradient descent.
        input_i, label_i = batch # collect input
        #output_i = self.forward(input_i)
        output_i = self.forward(input_i) # run input through the neural network
        loss = (output_i - label_i)**2 ## loss = squared residual
     
        return loss

def printResults(model,inputs):
    yhat_0 = model(inputs[0]) # answer is 0
    yhat_1 = model(inputs[1]) # answer is 1

    print(f"yhat_0:{yhat_0}/0.")
    print(f"yhat_1:{yhat_1}/1.")



def main():
    # Calling the LSTM Class
    model = CustomLSTM()
  
    useSGD = False
    if(useSGD):
        print("Using SGD")
    else:
        print("Using Batch GD")
    # Some training data and it's label
    inputs = torch.tensor([[0., 0.5, 0.25, 1.], [1., 0.5, 0.25, 1.]])
    labels = torch.tensor([0., 1.])

    #dataset = TensorDataset(inputs, labels) 
    #dataloader = DataLoader(dataset,batch_size=2)
    epochs = 2000
    optimizer = model.getOptimizer()

    print("Before training:")
    printResults(model,inputs)
    for i in range(0, epochs):
        # We need to invoke zero_grad() to prevent loss.backward() from 
        # accumulating the new gradient values with the ones from the previous 
        # step. Note that, each optimizer has 
        # its own way to use the gradients to update the weights.Nov 8, 20
        optimizer.zero_grad()
        # CAUTION: If gradient are not set to 0, it 
        # lowers the accuracy significantly

        # Note that data is in 2 lists
        totalLoss = torch.tensor(0.)
        for j in range(2): # For each input / for each batch
        # Can use dataloader which executes batch directly,
        # but we are doing it in a more basic way
        #for input,label in dataloader:
            loss = model.training_step((inputs[j],labels[j]))
            # loss value is a different dimension tensor as output is computed from the lstm function of pytorch
            totalLoss += loss[0] # Use 0 as loss is 1D vector, while totalLoss has no dimension
            if(useSGD):
                loss.backward()
                optimizer.step()
        # Calculate new weights
        print("*** totalLoss", totalLoss)
        #.backward is supported by each tensor object
        # will compute gradient whereever requires_grad is marked True
        if(not useSGD): # batch GD
            totalLoss.backward()  # Calculate the gradients or deltaW
            optimizer.step()      # Change weights or W <- W + deltaW

    # Note that prediction are still not that great as there is only a single sample
    print("After training:")
    printResults(model,inputs)

if __name__ == "__main__":
    main()
