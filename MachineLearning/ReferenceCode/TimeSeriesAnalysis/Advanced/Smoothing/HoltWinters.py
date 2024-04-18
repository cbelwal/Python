import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.holtwinters import ExponentialSmoothing


#--------- Define error/cost functions
def rmse(y, t):
  return np.sqrt(np.mean((y - t)**2))

def mae(y, t):
  return np.mean(np.abs(y - t))

def printErrorValues(res_hw, train, test, train_idx,test_idx,plotData=False):
    # Put values in same df
    df.loc[train_idx, 'PredHoltWinters'] = res_hw.fittedvalues
    df.loc[test_idx, 'PredHoltWinters'] = res_hw.forecast(N_test)

    if plotData:
       df[['Passengers', 'PredHoltWinters']].plot()
       plt.show()

    print("Train RMSE:", rmse(train['Passengers'], res_hw.fittedvalues))
    print("Test RMSE:", rmse(test['Passengers'], res_hw.forecast(N_test)))

    print("Train MAE:", mae(train['Passengers'], res_hw.fittedvalues))
    print("Test MAE:", mae(test['Passengers'], res_hw.forecast(N_test)))

df = pd.read_csv('./data/airline_passengers.csv', index_col='Month', parse_dates=True)

#----------------- Split on training and test
N_test = 15
train = df.iloc[:-N_test]
test = df.iloc[-N_test:]

train_idx = df.index <= train.index[-1]
test_idx = df.index > train.index[-1]

# The trend and seasonal components can be 'multiplicative' or 'additive'.
print("----------------- with add and add")
hw = ExponentialSmoothing(
    train['Passengers'],
    initialization_method='legacy-heuristic',
    trend='add', seasonal='add', seasonal_periods=12)
res_hw = hw.fit()

printErrorValues(res_hw,train,test,train_idx,test_idx,True)

#---------------- With different seasonal smoothing (mul rather than add)
print("----------------- with add and mul")
hw = ExponentialSmoothing(
    train['Passengers'],
    initialization_method='legacy-heuristic',
    trend='add', seasonal='mul', seasonal_periods=12)
res_hw = hw.fit()

# No need to plot
printErrorValues(res_hw,train,test,train_idx,test_idx)


print("----------------- with mul and mul")
hw = ExponentialSmoothing(
    train['Passengers'],
    initialization_method='legacy-heuristic',
    trend='add', seasonal='mul', seasonal_periods=12)
res_hw = hw.fit()

# No need to plot'
printErrorValues(res_hw,train,test,train_idx,test_idx)
