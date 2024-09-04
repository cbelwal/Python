# This is a way to build a LSTM unit from scratch
# Code is taken from Josh Stramer's video available here:
# https://www.youtube.com/watch?v=RHGiXPuo_pI

import torch # torch will allow us to create tensors.
import torch.nn as nn # torch.nn allows us to create a neural network.
import torch.nn.functional as F # nn.functional give us access to the activation and loss functions.
from torch.optim import Adam # optim contains many optimizers. This time we're using Adam

from torch.utils.data import TensorDataset, DataLoader # these are needed for the training data

'''
LSTM Basic Design (There are several other design variants available):

LT Memory       ----------------------+------------------------------ new LTM
  % to remember   | %potential to rem.|  pot. LTM                  |
                 Sig           Sig----x----tanh      Sig ----|   tanh
                  |             |           |         |      |     |
                blr1            blr2       bp1       bo1     |---- x
            Wlr1  +       Wpr1   +    Wp1   +     Wo1 +            |   
            *    Wlr2       *   Wpr2   *  Wp2     *  Wo2          |
            *     ~         *    ~     *   ~      *   ~           |-- new STM (yhat)
ST Memory   * ***********************************************
                  ~             ~           ~          ~
                  ~             ~           ~          ~
Input        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
             Stage 1              Stage 2                 Stage 3   

'''

# This is equivalent to the LSTMCell class in PyTorch
# Note that this is only applicable when feature size is 1
# and number of hidden states = 1, batch size = 1
# This does not include the linear layer which generaly resides outside the 
# LSTM cell
class LSTMCellFromScratch(nn.Module): # inherit from nn.Module
    def __init__(self):
        super().__init__() # Base class init  

    def initParams(self):
        torch.manual_seed(0) # So the same initial weights are used in every run

        # These will be used to create the normal distribution
        # NOTE: nn.LSTM used a uniform distribution (straight line) to init.weights
        mean = torch.tensor(0.0) # assign mean as tensor of value 0.0
        std = torch.tensor(1.0)  # assign std as tensor of value 1.0

        ## These are the Weights and Biases in the first stage, which determines what percentage
        ## of the long-term memory the LSTM unit will remember.
        # required_grad determines if weights need to updated via backpropagation
        # torch.normal: Returns a tensor of random numbers drawn from separate normal distributions whose mean and standard deviation are given.
        self.wlr1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.wlr2 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.blr1 = nn.Parameter(torch.tensor(0.), requires_grad=True)

        ## These are the Weights and Biases in the second stage, which determins the new
        ## potential long-term memory and what percentage will be remembered.
        self.wpr1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.wpr2 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.bpr1 = nn.Parameter(torch.tensor(0.), requires_grad=True)

        self.wp1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.wp2 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.bp1 = nn.Parameter(torch.tensor(0.), requires_grad=True)

        # These are the Weights and Biases in the third stage, which determines the
        ## new short-term memory and what percentage will be sent to the output.
        self.wo1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.wo2 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.bo1 = nn.Parameter(torch.tensor(0.), requires_grad=True)

    # stm: Short Term Memory
    # ltm: Long Term Memory
    def compute_lstm(self,input,stm,ltm):
        # *** Stage 1: what % of of LTM to remember
        ltm_remember_percent = torch.sigmoid((stm * self.wlr1) + 
                                              (input * self.wlr2) + 
                                              self.blr1)
        
        # *** Stage 2: Potention LTM and how much potential to add to current
        potential_ltm_percent = torch.sigmoid((stm * self.wpr1) + 
                                                   (input * self.wpr2) + 
                                                   self.bpr1)
        potential_memory = torch.tanh((stm * self.wp1) + 
                                      (input * self.wp2) + 
                                      self.bp1)
        
        # *** Make new LTM
        newLTM = ((ltm * ltm_remember_percent) + 
                       (potential_ltm_percent * potential_memory))
        
        # *** Stage 3: Compute new STM memory
        output_percent = torch.sigmoid((stm * self.wo1) + 
                                       (input * self.wo2) + 
                                       self.bo1)         
        newSTM = torch.tanh(newLTM) * output_percent

        return (newLTM,newSTM)
    
    # Here the input is a simple list with stock prices for each day
    # We will execute the LSTM after unrolling it
    def forward_unroll(self, inputs):
        ltm = 0 # long term memory is also called "cell state" and indexed with c0, c1, ..., cN
        stm = 0 # short term memory is also called "hidden state" and indexed with h0, h1, ..., cN
        
        day1 = inputs[0]
        day2 = inputs[1]
        day3 = inputs[2]
        day4 = inputs[3]
        
        ## Day 1
        ltm, stm = self.compute_lstm(day1, ltm, stm)
        
        ## Day 2
        ltm, stm = self.compute_lstm(day2, ltm, stm)
        
        ## Day 3
        ltm, stm = self.compute_lstm(day3, ltm, stm)
        
        ## Day 4
        ltm, stm = self.compute_lstm(day4, ltm, stm)
        
        ##### Now return short_memory, which is the 'output' of the LSTM.
        return stm
    
    # Standard forward, no unrolling
    def forward(self, inputs):
        ltm = 0 # long term memory is also called "cell state" and indexed with c0, c1, ..., cN
        stm = 0 # short term memory is also called "hidden state" and indexed with h0, h1, ..., cN
        
        for input in inputs:
            ltm, stm = self.compute_lstm(input, ltm, stm)        
        
        return stm

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
        output_i = self.forward_unroll(input_i) # run input through the neural network
        loss = (output_i - label_i)**2 ## loss = squared residual
     
        return loss

def printResults(model,inputs):
    yhat_0 = model(inputs[0]) # answer is 0
    yhat_1 = model(inputs[1]) # answer is 1

    print(f"yhat_0:{yhat_0}/0.")
    print(f"yhat_1:{yhat_1}/1.")



def main():
    # Calling the LSTM Class
    model = LSTMCellFromScratch()
    model.initParams()

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
        # Note that data is in 2 lists
        optimizer.zero_grad() # Better results using this.
        totalLoss = torch.tensor(0.)
        for j in range(2): # For each input / for each batch
        # Can use dataloader which executes batch directly,
        # but we are doing it in a more basic way
        #for input,label in dataloader:
            loss = model.training_step((inputs[j],labels[j]))
            totalLoss += loss
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
