import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.graphics.tsaplots as sgt
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats.distributions import chi2
import statsmodels.tsa.stattools as sts 


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

# Use PACF plot to analyze coefficients for AR

#------------------- Find ARMA 1
model_ar_1 = ARIMA(df.market_value, order=(1,0,0)) #p=1, is required for AR
results_ar_1 = model_ar_1.fit()
print(results_ar_1.summary())

#------------------- Find ARMA 2
model_ar_2 = ARIMA(df.market_value, order=(2,0,0)) #p=2, is required for ARMA
results_ar_2 = model_ar_2.fit()
print(results_ar_2.summary())
#In Summary P > |z| should be lower, the lower the better

#Define a Log Likelihood Ratio (LLR) test to compare the models
def LLR_test(mod_1, mod_2, DF=1):
    L1 = mod_1.fit().llf
    L2 = mod_2.fit().llf
    LR = (2*(L2-L1))
    p = chi2.sf(LR, DF).round(3)
    return p

print("LLR Test:",LLR_test(model_ar_1, model_ar_2)) #0.0 is considered high value

#Run ADFuller Test
print("AD Fuller Results:",sts.adfuller(df.market_value))

#Normalize the time series, how much % of first value