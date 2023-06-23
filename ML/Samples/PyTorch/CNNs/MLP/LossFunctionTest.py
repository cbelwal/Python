import torch
import torch.nn as nn
import numpy as np
import SoftMaxTest
import math


def CrossEntropyPyTorch(values, actualProb):
    tensorValues = torch.FloatTensor(values)
    tensorActualProb = torch.LongTensor(actualClass) 
    criterion = nn.CrossEntropyLoss() #LogSoftMax + NNLoss
    loss = criterion(tensorValues, tensorActualProb)
    return loss

def CrossEntropyManual_PyTorch_NNLoss(values, actualProb): #TODO
    #Compute LogSoftMax
    #logSoftMaxValues = SoftMaxTest.logSoftmaxPyTorch(values)
    #value = np.log(values)
    tensor = torch.FloatTensor(values) 
    tensorValues = nn.LogSoftmax()(tensor)
    #Apply NNLoss
    criterion = nn.NLLLoss()
    #tensorValues = torch.FloatTensor(logSoftMaxValues)
    tensorActualProb = torch.LongTensor(actualProb)
    loss = criterion(tensorValues, tensorActualProb)
    return loss
    #den = 0.
    #for i in range(len(softMax)):
     #   den += e ** numList[i]
    
if __name__ == '__main__':
    values = [[.03, .4, .5]] 
    actualProb = [1,0,0]
    actualClass = [0]

    print("Cross Entropy from PyTorch:",CrossEntropyPyTorch(values,actualClass))
    print("Cross Entropy from Manual_PyTorch_NNLoss:",CrossEntropyManual_PyTorch_NNLoss(values,actualClass))
    #print("Softmax from Manual:",softmaxManual(values))