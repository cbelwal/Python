import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('./data/airline_passengers.csv', index_col='Month', parse_dates=True)
df.index.freq = 'MS'

# The difference data should show stationarity 
df['1stdiff'] = df['Passengers'].diff()

#df.plot(figsize=(10, 5))
#plt.show()

df['LogPassengers'] = np.log(df['Passengers'])

from statsmodels.tsa.arima.model import ARIMA

# Define the window size for train/test
Ntest = 12 
# df shape is (144,3)
# Store everything from start till Ntest elements from end
# so train shape become (132,3)
train = df.iloc[:-Ntest]
# store Ntest elements from the end
# shape will be (12,3) 
test = df.iloc[-Ntest:] 

train_idx = df.index <= train.index[-1]
test_idx = df.index > train.index[-1]

arima = ARIMA(train['Passengers'], order=(1,0,0)) #Only AR with p=1
arima_result = arima.fit()

# Predict on train data
# There is no concept of fittedvalues in ARIMA class
# hence we have to predict on the training data to get the fitted values
df.loc[train_idx, 'AR(1)'] = arima_result.predict(
    start=train.index[0], end=train.index[-1])

# Now forecast the values
# Forcecast for next Ntest values 
prediction_result = arima_result.get_forecast(Ntest)

forecast = prediction_result.predicted_mean # This property contains the pred. values
df.loc[test_idx, 'AR(1)'] = forecast

# Plot predictions with actual, include both predicted and actual
#df[['Passengers', 'AR(1)']].plot(figsize=(10, 5))
#plt.show()

# Get more details on prediction result, including confidence interval
print("Confidence Intervals",prediction_result.conf_int())


# Function will take a model object and plot the predicted, forecasted
# values and also shade the confidence interval
def plot_fit_and_forecast(result,title=""):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df['Passengers'], label='data')

    # plot the curve fitted on train set
    train_pred = result.fittedvalues
    ax.plot(train.index, train_pred, color='green', label='fitted')

    # forecast the test set
    prediction_result = result.get_forecast(Ntest)
    conf_int = prediction_result.conf_int()
    # Note the format of the fiels, it is prefixed with lower and upper
    lower, upper = conf_int['lower Passengers'], conf_int['upper Passengers']
    forecast = prediction_result.predicted_mean
    ax.plot(test.index, forecast, label='forecast')
    ax.fill_between(test.index, \
                    lower, upper, \
                  color='red', alpha=0.3)
    ax.legend()
    plt.title(title)
    plt.show()

# The plot the previous result with AR(1)
plot_fit_and_forecast(arima_result,"AR(1)")

#Create new model with AR(10)----------
arima = ARIMA(train['Passengers'], order=(10,0,0)) 
arima_result = arima.fit()
plot_fit_and_forecast(arima_result,"AR(10)")


#Create new model with MA(1)----------
arima = ARIMA(train['Passengers'], order=(0,0,1)) 
arima_result = arima.fit()
plot_fit_and_forecast(arima_result, "MA(1)")