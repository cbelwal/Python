#Vector Autoregressive (VAR)  Model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.graphics.tsaplots as sgt
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats.distributions import chi2
import statsmodels.tsa.stattools as sts 
from statsmodels.tsa.api import VAR
from arch import arch_model


#Load Data
df_raw = pd.read_csv("./Data/index2018.csv")

#Data Sanitization--------------
df_copy = df_raw.copy()
df_copy.date = pd.to_datetime(df_copy.date, dayfirst = True)
df_copy.set_index("date", inplace=True)
df_copy=df_copy.asfreq('b')
df_copy=df_copy.fillna(method='ffill')
df_copy['market_value']=df_copy.spx


size = int(len(df_copy)*0.8)
df, df_test = df_copy.iloc[:size], df_copy.iloc[size:]
#---------------------------------------

#Define all columns that need to considered int he VectorAR model
df_all = df[['spx', 'dax', 'ftse', 'nikkei']][1:]

model_var_ret = VAR(df_all)
model_var_ret.select_order(20) #Number of coeffs, 4 variables, so 20/4 = 5 coeffs
results_var_ret = model_var_ret.fit(ic = 'aic')

print(results_var_ret.summary())

#Forecast 1000 values directly
results_var_ret.plot_forecast(1000)
plt.show()

start_date = "2014-07-15"
end_date = "2015-01-01"

lag_order_ret = results_var_ret.k_ar
var_pred_ret = results_var_ret.forecast(df_all.values[-lag_order_ret:], len(df_test[start_date:end_date]))

df_ret_pred = pd.DataFrame(data = var_pred_ret, index = df_test[start_date:end_date].index,
                                columns = df_test[start_date:end_date].columns[4:8])

df_ret_pred.ret_nikkei[start_date:end_date].plot(figsize = (10,5), color = "red")
df_test.ret_nikkei[start_date:end_date].plot(color = "blue")

plt.title("Real vs Prediction", size = 24)
plt.show()