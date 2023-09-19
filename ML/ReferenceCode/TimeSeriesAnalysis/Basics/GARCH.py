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
model_garch_1_1 = arch_model(df.returns[1:], mean = "Constant", vol = "GARCH", p = 1, q = 1)
results_garch_1_1 = model_garch_1_1.fit(update_freq = 5)
results_garch_1_1.summary()


#----------------- Higher lag GARCH Models
model_garch_1_2 = arch_model(df.returns[1:], mean = "Constant",  vol = "GARCH", p = 1, q = 2)
results_garch_1_2 = model_garch_1_2.fit(update_freq = 5)
results_garch_1_2.summary()


