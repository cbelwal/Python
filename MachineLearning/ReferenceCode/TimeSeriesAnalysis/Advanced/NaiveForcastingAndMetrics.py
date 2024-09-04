import numpy as np
import pandas as pd


from sklearn.metrics import mean_absolute_percentage_error, \
  mean_absolute_error, r2_score, mean_squared_error

# Load data set
df = pd.read_csv('./data/SPY.csv', index_col='Date', parse_dates=True)

# Set predicted price as next step Close price
# This is done by the pandas shift operand
df['ClosePrediction'] = df['Close'].shift(1)

#Assign to different variables, ignore the first value
y_true = df.iloc[1:]['Close']
y_pred = df.iloc[1:]['ClosePrediction']

#----------------- Compute some error metrics
# Compute SSE
# .dot is used for array, will multiply and sum the product 
sse = (y_true - y_pred).dot(y_true - y_pred)
manual_mse = sse/len(y_true)
manual_rmse = np.sqrt(manual_mse)

print("ss,manual_mse,manual_rmse:",sse,manual_mse,manual_rmse)

mse = mean_squared_error(y_true, y_pred)
print("mse",mse)

# MAE
mae = mean_absolute_error(y_true, y_pred)
print("mae",mae)

# R2 error,  0(bad) - 1(good) range, R2 = 0, prediction is mean
r2 = r2_score(y_true, y_pred)
print("R2 Score",r2)

# MAPE
# Symmetric Mean Absolute Percentage Error
mape = mean_absolute_percentage_error(y_true, y_pred)
print("MAPE:",mape)

# sMAPE
# Symmetric Mean Absolute Percentage Error
def smape(y_true, y_pred):
  numerator = np.abs(y_true - y_pred)
  denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
  ratio = numerator / denominator
  return ratio.mean()

smape = smape(y_true, y_pred)
print("SMape",smape)

