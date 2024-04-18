#a[start:end:step]

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pmdarima as pm

df = pd.read_csv('./data/airline_passengers.csv', index_col='Month', parse_dates=True)
df.index.freq = 'MS'

# Assign Log here before the train/test split
df['LogPassengers'] = np.log(df['Passengers'])
# Define the window size for train/test
Ntest = 12 
# df shape is (144,3)
# Store everything from start till Ntest elements from end
# so train shape become (132,3)
train = df.iloc[:-Ntest]
# store Ntest elements from the end
# shape will be (12,3) 
test = df.iloc[-Ntest:] 


#----------- Run Auto ARIMA model
model = pm.auto_arima(train['Passengers'],
                      trace=True,
                      suppress_warnings=True,
                      seasonal=True, m=12)

print(model.summary())

# Get Test predictions and confidence intervals
test_pred, confint = model.predict(n_periods=Ntest, return_conf_int=True)

#----- Plot Test Data
# Show the data, show actual test data and then predicted, with confidence interval
# in between 
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(test.index, test['Passengers'], label='data')
ax.plot(test.index, test_pred, label='forecast')
ax.fill_between(test.index, \
                confint[:,0], confint[:,1], \
                color='red', alpha=0.3)
ax.legend()
plt.title("Test data, Test Predictions and confidence intervals")
plt.show()

#----- Plot Train, Test Data along with predictions and confidence intervals on the test data
train_pred = model.predict_in_sample(start=0, end=-1)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df.index, df['Passengers'], label='data')
ax.plot(train.index, train_pred, label='fitted')
ax.plot(test.index, test_pred, label='forecast')
ax.fill_between(test.index, \
                confint[:,0], confint[:,1], \
                color='red', alpha=0.3)
ax.legend()
plt.title("Train and Test data, Train and  Test Predictions and confidence intervals")
plt.show()

#------------------- Run on LOG data


model = pm.auto_arima(train['LogPassengers'],
                      trace=True,
                      suppress_warnings=True,
                      seasonal=True, m=12)

print(model.summary())

# Get Test predictions and confidence intervals
test_pred_log, confint = model.predict(n_periods=Ntest, return_conf_int=True)

#----- Plot Test Data
# Show the data, show actual test data and then predicted, with confidence interval
# in between 
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(test.index, test['LogPassengers'], label='data')
ax.plot(test.index, test_pred_log, label='forecast')
ax.fill_between(test.index, \
                confint[:,0], confint[:,1], \
                color='red', alpha=0.3)
ax.legend()
plt.title("LogPassengers: Test data, Test Predictions and confidence intervals")
plt.show()

#----- Plot Train, Test Data along with predictions and confidence intervals on the test data
train_pred = model.predict_in_sample(start=0, end=-1)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df.index, df['LogPassengers'], label='data')
ax.plot(train.index, train_pred, label='fitted')
ax.plot(test.index, test_pred_log, label='forecast')
ax.fill_between(test.index, \
                confint[:,0], confint[:,1], \
                color='red', alpha=0.3)
ax.legend()
plt.title("LogPassengers: Train and Test data, Train and  Test Predictions and confidence intervals")
plt.show()

### forecast RMSE
def rmse(t, y):
  return np.sqrt(np.mean((t - y)**2))

print("Non-logged RMSE:", rmse(test['Passengers'], test_pred))
print("Logged RMSE:", rmse(test['Passengers'], np.exp(test_pred_log)))


# ----------------- Build Auto ARIMA by specifying max values
### non-seasonal
model = pm.auto_arima(train['LogPassengers'],
                      trace=True,
                      suppress_warnings=True,
                      d=0,
                      max_p=12, max_q=2, max_order=14,
                      stepwise=True, # stepwise = True, - Dont do an exhaustive search.False, - Do an exhaustive search.
                      seasonal=False)
