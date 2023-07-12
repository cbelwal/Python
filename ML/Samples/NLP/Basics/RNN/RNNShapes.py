import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.layers import Input, SimpleRNN, Dense, Flatten
from keras.models import Model
from keras.optimizers import SGD, Adam
import keras

from datetime import datetime

import tensorboard

# N = number of samples
# T = sequence length
# D = number of input features
# M = number of hidden units
# K = number of output unit

# Some data
N = 1
T = 10
D = 3
K = 2
X = np.random.randn(N,T,D) #Random data, creates a TxD matrix.

#Make an RNN
M = 5 #number of hidden units
i = Input(shape=(T,D))
x = SimpleRNN(M)(i)
x = Dense(K)(x)

model = Model(i,x)

# Compile and fit
# At this point no weights are trained so this will be dummy output
out = model.predict(X) #X is random
print(out) #---------- Output #1 

print("Model Summary:", model.summary())

# ------------ layer 1 weights
Wx,Wh,bh = model.layers[1].get_weights()
#print("Model Weights,","Wx:",Wx,"Wh:",Wh,"bh:",bh) #bh is bias term
print("Shapes of layer 1 weights:", "Wx:",Wx.shape,"Wh:",Wh.shape,"bh:",bh.shape)

# ---- layer 2 weights
Wo, bo = model.layers[2].get_weights()
print("Shapes of layer 2 weights:", "Wo:",Wo.shape,"bo:",bo.shape)

# Define the Keras TensorBoard callback.
logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")

#tensorboard details here: 
# https://medium.com/aiguys/visualizing-deep-learning-model-architecture-5c18e057b73e
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

h_last = np.zeros(M) #initial hidden state, since M = 5 values
x = X[0] # 1 sample
Yhats = [] # store outputs

# Manually execute the RNN steps, that the model.predict runs automatically
# x has size 10x3
for t in range(T):
    # h_last is also fully connected, better visualized through rolling/rollout of hidden layer
    h = np.tanh(x[t].dot(Wx) + h_last.dot(Wh) + bh) #Output form hidden RNN layer
    y = h.dot(Wo) + bo #Final output, matches
    Yhats.append(y)

    h_last = h #This is a RNN to needed to use previous value

# this is the final value
print(Yhats[-1])  #---------- This should match the output give in Output #1