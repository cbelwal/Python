# This is code for stock proice prediction using LSTM and 
# also attention
#
# This code is a modified version of code found in:
# https://drlee.io/revolutionizing-time-series-prediction-with-lstm-with-the-attention-mechanism-090833a19af9


# ----------- Libraries
import yfinance as yf
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler


# ---------- Data Prep
# Download GOOG stock data, between specified dates
# data is a pandas df
data = yf.download('GOOG', start='2020-01-01', end='2024-03-01')

# Use the 'Close' price for prediction
close_prices = data['Close'].values

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1)) # This will convert all values to between 0 and 1
# MinMaxScaler is useful when you want to ensure that all features have 
# a similar scale and are constrained within a specific range, which can 
# improve the performance of algorithms that are sensitive to the scale of input features, 
# such as gradient-based optimization algorithms.
close_prices_scaled = scaler.fit_transform(close_prices.reshape(-1, 1))