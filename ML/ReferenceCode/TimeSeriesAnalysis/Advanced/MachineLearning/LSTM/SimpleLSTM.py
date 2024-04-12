# In this we only look at the values and inputs of the LSTM unit
# The full code has not been incorporated

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, GlobalMaxPooling1D, LSTM, GRU
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint

T = 12 # Number of time steps
# Input is Number of time steps, and each time step has 1 value
i = Input(shape=(T, 1)) 
# Set LSTM layer with 24 units
# Here units is a misleading defincition, it shows how many LSTM units are there in the layer
# For the code give below: x = LSTM(2)(i), the neural net will look like
# 
#    / LSTM1
#  x 
#    \ LSTM2
#
# If T = 2, then the rolled out LSTM will look like:
#
#    / LSTM1 - LSTM1 -
#  x        \/      
#    \ LSTM2 - LSTM2 - 
#

 
x = LSTM(2)(i) 
x = Dense(1)(x)
model = Model(i, x)
