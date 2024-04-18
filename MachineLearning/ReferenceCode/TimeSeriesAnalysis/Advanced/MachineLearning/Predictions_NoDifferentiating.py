# Predict using different ML Models
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define the models to be used
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error, r2_score

# Read dataset and create DF
df = pd.read_csv('./data/airline_passengers.csv', index_col='Month', parse_dates=True)
df['LogPassengers'] = np.log(df['Passengers'])

# Break into train and test
Ntest = 12
train = df.iloc[:-Ntest] # All except the last 12 points
test = df.iloc[-Ntest:]


series = df['LogPassengers'].to_numpy() # Convert to numpy array for easier handling

# Convert to a regression problem
# Use past X values to predict Y
# Y = X at time T+11
T = 10 # Use 10 past values for each data point
X = []
Y = []

for t in range(len(series) - T):
  x = series[t:t+T] # Store 10 values in x
  X.append(x)
  y = series[t+T] # Y is single value 
  Y.append(y)

X = np.array(X).reshape(-1, T) # Reshape the array to specified dimensions
Y = np.array(Y)
N = len(X)

print(f"XShape {X.shape}, Y Shape {Y.shape}")

# Assign Train and Test test based on modified data
Xtrain, Ytrain = X[:-Ntest], Y[:-Ntest] #X is an array of multiple values
Xtest, Ytest = X[-Ntest:], Y[-Ntest:]

#-------------- Linear Regression
lr = LinearRegression()
lr.fit(Xtrain, Ytrain)
print(f"Train Score:{lr.score(Xtrain, Ytrain)}") # .score is the R^2 metric
# Calling score will run the computations automatically
print(f"Test Score:{lr.score(Xtest, Ytest)}") # .score is the R^2 metric

# Create a Boolean index
train_idx = df.index <= train.index[-1] # train.index[-1] is the last index
test_idx = ~train_idx # All other indexes will be False
train_idx[:T] = False # first T values are not predictable, so mark it as False

# 1-step forecast for train and test
df.loc[train_idx, 'LR_1step_train'] = lr.predict(Xtrain) 
df.loc[test_idx, 'LR_1step_test'] = lr.predict(Xtest)

# Plot with Log values and their forecast
df[['LogPassengers', 'LR_1step_train', 'LR_1step_test']].plot(figsize=(10, 5))
plt.title("Plot with single step predicted values")
plt.show()


# Multi-step Forecast
multistep_predictions = []

# first test input
# XTest is an array of multiple values so XTest[0] will have Ntest(10) values
last_x = Xtest[0] 

# This essentially makes predictions using last predicted
# value. Using np.roll() we move first value to last and then replace
# the last value with previously predicted value  
while len(multistep_predictions) < Ntest:
  p = lr.predict(last_x.reshape(1, -1))[0]
  
  # update the predictions list
  multistep_predictions.append(p)
  
  # make the new input
  # np.roll will move the beginning values to the end
  # Example: if A = [1,2,3,4,5]
  # np.roll(A,2) make A = [3,4,5,1,2]
  # The -1 shifts it to the left
  # Example: if A = [1,2,3,4,5]
  # np.roll(A,-1) make A = [2,3,4,5,1]
  last_x = np.roll(last_x, -1)
  last_x[-1] = p

df.loc[test_idx, 'LR_multistep'] = multistep_predictions   

# plot 1-step and multi-step forecast
df[['LogPassengers', 'LR_multistep', 'LR_1step_test']].plot(figsize=(10, 5))
plt.title("Plot with multi-step predicted values, Train is not shown")
plt.show()

# Now we make a multi-output dataset
# For each prediction NTest values will be predicted
Tx = T
Ty = Ntest
X = []
Y = []
for t in range(len(series) - Tx - Ty + 1):
  x = series[t:t+Tx]
  X.append(x)
  y = series[t+Tx:t+Tx+Ty]
  Y.append(y)

X = np.array(X).reshape(-1, Tx)
Y = np.array(Y).reshape(-1, Ty)
N = len(X)
print("X.shape", X.shape, "Y.shape", Y.shape)

Xtrain_m, Ytrain_m = X[:-1], Y[:-1]
print("Xtrain_m.shape", X.shape, "Ytrain_m.shape", Y.shape)

Xtest_m, Ytest_m = X[-1:], Y[-1:]

lr = LinearRegression()
lr.fit(Xtrain_m, Ytrain_m)
print(f"Train SST = {lr.score(Xtrain_m, Ytrain_m)}") # Sum of Squares
print(f"Test SST = {lr.score(Xtest_m, Ytest_m)}")

