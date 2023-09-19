#Generalized Autoregressive Conditional Heteroskedasticity (GARCH) model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.graphics.tsaplots as sgt
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats.distributions import chi2
import statsmodels.tsa.stattools as sts 

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

del df_copy['spx']
del df_copy['dax']
del df_copy['ftse']
del df_copy['nikkei']
size = int(len(df_copy)*0.8)
df, df_test = df_copy.iloc[:size], df_copy.iloc[size:]
#---------------------------------------

#Compute Returns as % change
df['returns'] = df.market_value.pct_change(1)*100


#Create squared returns
df['sq_returns'] = df.returns.mul(df.returns)


#Squared returns show the volatility
df.sq_returns.plot(figsize=(10,5))
plt.title("Volatility", size = 24)
plt.show()

#------------------- GARCH Model
start_date = "2014-07-15"
end_date = "2015-01-01"

mod_garch = arch_model(df_copy.market_value[1:], vol = "GARCH", p = 1, q = 1, mean = "constant", dist = "Normal")
res_garch = mod_garch.fit(last_obs = start_date, update_freq = 10)

#prediction vs forecast: 
# forecast is done on future, prediction is when we have current data

pred_garch = res_garch.forecast(horizon = 1, align = 'target')

pred_garch.residual_variance[start_date:].plot(figsize = (20,5), color = "red", zorder = 2)
df_test.ret_ftse.abs().plot(color = "blue", zorder = 1)
plt.title("Volatility Predictions", size = 24)
plt.show()


