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

#del df_copy['spx']
#del df_copy['dax']
#del df_copy['ftse']
del df_copy['nikkei']
size = int(len(df_copy)*0.8)
df, df_test = df_copy.iloc[:size], df_copy.iloc[size:]

start_date = "2014-07-15"
end_date = "2015-01-01"

from pmdarima.arima import auto_arima

# The (p,d,q) order of the model for the autoregressive, differences, and moving
model_auto = auto_arima(df.market_value[1:], exogenous = df[['spx', 'dax', 'ftse']][1:],
                       m = 5, max_p = 5, max_q = 5, max_P = 5, max_Q = 5)

df_auto_pred = pd.DataFrame(model_auto.predict(n_periods = len(df_test[start_date:end_date]),
                            exogenous = df_test[['spx', 'dax', 'ftse']][start_date:end_date]),
                            index = df_test[start_date:end_date].index)

df_auto_pred.plot(figsize = (10,5), color = "red")
df_test.ret_ftse[start_date:end_date].plot(color = "blue")
plt.title("Auto Model Predictions vs Real Data", size = 24)
plt.show()

