# This code implements Simple Exponential Smoothing (SES)
# non-Simple Exponential Smoothing is Holt-Winters and done in a separate Python file
# Holt Winters forecast uses EWMA for forcasting purposes
# y_hat_t+1 = 	α.y_t + (1 - α).y_hat_t
# y_hat_t = prediction at time t

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

df = pd.read_csv('./data/airline_passengers.csv', index_col='Month', parse_dates=True)

# Data is passed on constructor
ses = SimpleExpSmoothing(df['Passengers'])
# This will result in some errors

# Set frequency to months
df.index.freq = 'MS'

# Reassign ses after adding freq.
ses = SimpleExpSmoothing(
    df['Passengers'],
    initialization_method='legacy-heuristic')

alpha = 0.2

res = ses.fit(smoothing_level=alpha, optimized=False)

# Predict from start to end on the data used for training
df['SES_Pred_AlphaGiven'] = res.predict(start=df.index[0], end=df.index[-1])

# Check if all predicted values are close within specific tolerance, of the fitted values
print("All Close:",np.allclose(df['SES_Pred_AlphaGiven'], res.fittedvalues,atol=1e-08))

# -- Plot Predicted and actual values
#df.plot()
#plt.show()

# Now fit data gain but dont give value of Alpha
# This will make the model predict the value of alpha that leads to lowers errors
res = ses.fit()
# Predict from start to end on the data used for training
df['SES_Pred_AlphaComputed'] = res.predict(start=df.index[0], end=df.index[-1])
# -- Plot Predicted and actual values
# Alpha computed is very close to the give values
#df.plot()
#plt.show()


#---------------- Now do same with breaking into training and test sets
N_test = 15
train = df.iloc[:-N_test]
test = df.iloc[-N_test:]

ses = SimpleExpSmoothing(
    train['Passengers'],
    initialization_method='legacy-heuristic')
res = ses.fit()
df['SES_fitted'] = res.fittedvalues
# Forecast N_test time steps form the point where current data finishes
df['SES_forecast'] = res.forecast(N_test)
df[['Passengers','SES_fitted','SES_forecast']].plot()
plt.show()

'''
# There is another way to do this, by merging into same column
# This way we do not have to use two extra columns 'SES_fitted' and'SES_forecast'
# boolean series to index df rows
train_idx = df.index <= train.index[-1]
test_idx = df.index > train.index[-1]

df.loc[train_idx, 'SESfitted'] = res.fittedvalues
df.loc[test_idx, 'SESfitted'] = res.forecast(N_test)
df[['Passengers', 'SESfitted']].plot();

'''



