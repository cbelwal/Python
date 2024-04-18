import torch
import torch.nn as nn
import numpy as np
import math

def softmaxNumPy(numList):
    npArray = np.array(numList) 
    return np.exp(npArray) / sum(np.exp(npArray))

def softmaxPyTorch(numList):
    tensor = torch.FloatTensor(numList) 
    sm = nn.Softmax()(tensor)
    return sm# nn.functional.softmax(tensor,dim=0)

def logSoftmaxPyTorch(numList):
    tensor = torch.FloatTensor(numList) 
    sm = nn.LogSoftmax()(tensor)
    return sm# nn.functional.softmax(tensor,dim=0)

def logSoftmaxManual(numList):
    e = 2.71828
    softMax = softmaxManual(numList)
    logSoftMaxValues = [] #np.log(softMaxValues)
    for v in softMax:
        logSoftMaxValues.append(math.log(v,e)) 
    return logSoftMaxValues

def softmaxManual(numList):
    e = 2.71828
    den = 0.
    for i in range(len(numList)):
        den += e ** numList[i]
    values =[]
    for i in range(len(numList)):
        values.append((e ** numList[i])/den)
    return values

if __name__ == '__main__':
    values = [-.03, .4, .5]
    print("Softmax from NumPy:",softmaxNumPy(values))
    print("Softmax from PyTorch:",softmaxPyTorch(values))
    print("Softmax from Manual:",softmaxManual(values))
    print("******")
    print("LogSoftmax from PyTorch:",logSoftmaxPyTorch(values))
    print("LogSoftmax from Manual:",logSoftmaxManual(values))
