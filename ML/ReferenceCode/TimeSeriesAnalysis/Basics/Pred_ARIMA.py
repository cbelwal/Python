import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.graphics.tsaplots as sgt
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats.distributions import chi2
import statsmodels.tsa.stattools as sts 


#Load Data
#This data is downloaded from Yahoo Finance using the yfinance library
df_raw = pd.read_csv("./Data/index2018.csv")

#Data Sanitization--------------
df_copy = df_raw.copy()
df_copy.date = pd.to_datetime(df_copy.date, dayfirst = True)
df_copy.set_index("date", inplace=True)
df_copy=df_copy.asfreq('b')
df_copy=df_copy.fillna(method='ffill')
df_copy['market_value']=df_copy.spx

del df_copy['spx']
del df_copy['dax']
del df_copy['ftse']
del df_copy['nikkei']
size = int(len(df_copy)*0.8)
df, df_test = df_copy.iloc[:size], df_copy.iloc[size:]

# The (p,d,q) order of the model for the autoregressive, differences, and moving
model_arima_1_1 = ARIMA(df.market_value, order=(3,1,2)) #ARIMA 1,1,1
results_arima_1_1 = model_arima_1_1.fit()
print(results_arima_1_1.summary())

start_date = "2014-07-15"
end_date = "2015-01-01"

#end_date = "2019-10-23"
df_pred = results_arima_1_1.predict(start = start_date, end = end_date)

df_pred[start_date:end_date].plot(figsize = (10,5), color = "red")
df_test.market_value[start_date:end_date].plot(color = "blue")
plt.title("Predictions vs Actual", size = 24)
plt.show()


