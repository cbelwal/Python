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

# Shape is r*c format
print("Current Shape,",close_prices.shape)
# -1 is like the *, will set the row or column to whatever shape that is needed
close_prices = close_prices.reshape(-1, 1) # Use values.reshape in later versions 
print("After ReShape,",close_prices.shape)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1)) # This will convert all values to between 0 and 1
# MinMaxScaler is useful when you want to ensure that all features have 
# a similar scale and are constrained within a specific range, which can 
# improve the performance of algorithms that are sensitive to the scale of input features, 
# such as gradient-based optimization algorithms.
close_prices_scaled = scaler.fit_transform(close_prices)

print("close_prices shape: ", close_prices_scaled.shape)
X = close_prices_scaled[:-1] # All except last
print("X Shape: ", X.shape)
y = close_prices_scaled[1:] # All except first 
print("Y Shape: ", y.shape)


# Pytorch LSTM require the following shape for Input:
# input: tensor of shape (L,H) for unbatched input, 
#
# (L,N,H)when batch_first=False
# (N,L,H)when batch_first=True
#
# or containing the features of the input sequence. The input can also be a packed variable length sequence.
#
# L = sequence length
# N = batch size 
# H = input_size / feature dimensions
#
# Reshape for LSTM

X = X.reshape(-1, 1, 1)
print("After reshape for LSTM input, X Shape: ", X.shape) #LSTM X Shape:  (1046, 1, 1)
y = y.reshape(-1, 1)
print("After reshape for LSTM input,y Shape: ", y.shape)  #LSTM y Shape:  (1046, 1)

# NOTE: Since in the LSTM we set batch_first=True, the dimension of input are:
# (N,L,H), or (batch_size, sequence length, input size)
# For Shape (1046, 1, 1), there are 1046 batches, seq_length = 1, input size = 1
# 
# So essentially here we going to run 1046 independent sequences, with seq length of 1
# The prediction is only for time t, using the prior value of t-1 so number of
# y_hat will depend on the prediction  

# Do Train-test split with 20% test
# random_state: controls the shuffling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=36)

# Convert df to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Create the LSTM Class
# LSTM with Attention Mechanism
class LSTMWithAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, num_layers=1):
        super(LSTMWithAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # Since batch_first=True, input_dim should have 3 dimensions 
        # with number of batches in 1st dimension
        # Since our batch size is entire input, entire flow sequence will be completed
        # before weights are updated
        # 
        # 
        # LSTM -<
        # 
        #  
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True) 
        self.attention = nn.Linear(in_features=hidden_dim,out_features=1)
        self.fc = nn.Linear(in_features=hidden_dim, out_features=output_dim)

    def forward(self, x):
        #print("x Shape",x.shape) # ([836, 1, 1]): 836 batch size, seq length = 1, num features = 1
        lstm_out, _ = self.lstm(x) # pass through the LSTM
        #print("lstm_out Shape",lstm_out.shape) # ([836, 1, 50]): 836 batch size, seq length = 1, num features = 50 (hidden size)
        attention_l1 = self.attention(lstm_out) # linear layer
        #print("attention_l1 Shape ",attention_l1.shape) # ([836, 1, 1])
        # squeeze(): Returns a tensor with all specified dimensions of input of size 1 removed.  
        # For example, if input is of shape:  
        # (A×1×B×C×1×D) then the input.squeeze() will be of shape:  (A×B×C×D).
        # -1 is the last dimension, so the num_feature dimension will be dropped
        # Good explanation of squeeze() and unsqueeze(): https://stackoverflow.com/questions/61598771/squeeze-vs-unsqueeze-in-pytorch
        attention_l1_squeezed = attention_l1.squeeze(dim=-1)
        #print("attention_l1 Squeezed Shape ",attention_l1_squeezed.shape) # torch.Size([836, 1])
        # dim (int) – A dimension along which Softmax will be computed (so every slice along dim will sum to 1).
        # NOTE: Since there is only 1 value going into the Softmax, output of the Softmax
        #       is going to be 1
        attention_weights = torch.softmax(attention_l1_squeezed, dim=-1)
        # print("attention_weight Shape",attention_weights.shape) # ([836, 1]): 836 batch size, output_size = 1
        #
        # unsqueeze(): Returns a new tensor with a dimension of size one inserted at the specified position.
        # The returned tensor shares the same underlying data with this tensor.
        attention_weights_unsqueezed = attention_weights.unsqueeze(-1)
        #print("attention_weights_unsqueezed Shape",attention_weights_unsqueezed.shape) # ([836, 1, 1])
        # due to unsqueeze both lstm_out and attention_weights_unsqueezed are same shape
        context_vector = torch.sum(lstm_out * attention_weights_unsqueezed, dim=1)
        print("context_vector Shape",context_vector.shape) # ([836, 50]) : 836 batch size, output_size = 1
        out = self.fc(context_vector) # pass context_vector through FC layer which give 1 output
        return out # out is a scalar
    
# Setup for Model training
model = LSTMWithAttention(input_dim=1, hidden_dim=50) # 50 input feature for better
criterion = nn.MSELoss() # This is a regression problem so use MSE
optimizer = torch.optim.Adam(model.parameters(), lr=0.01) # Default lr is .001, use .01


# Training loop
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train_tensor)
    loss = criterion(output, y_train_tensor)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0: # Run against test every 10 epochs
        model.eval()
        test_pred = model(X_test_tensor)
        test_loss = criterion(test_pred, y_test_tensor)
        print(f'Epoch {epoch}, Loss: {loss.item()}, Test Loss: {test_loss.item()}')

# Predictions
model.eval()
yhat = model(X_test_tensor).detach().numpy()
yhat_actual = scaler.inverse_transform(yhat) #scalet is MinMaxScaler object

# Plotting
plt.figure(figsize=(15, 5))
plt.plot(scaler.inverse_transform(y_test), label='Actual')
plt.plot(yhat_actual, label='Predicted')
plt.title('Stock Price Prediction')
plt.legend()
plt.show()

# Calculate MSE
mse = mean_squared_error(scaler.inverse_transform(y_test), yhat_actual)
print(f'Mean Squared Error for test: {mse}')