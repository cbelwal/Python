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
#---------------------------------------

# Use ACF plot to analyze coefficients for MA

#------------------- Find MA 1
# The (p,d,q) order of the model for the autoregressive, differences, and moving
model_arima_1_1 = ARIMA(df.market_value, order=(1,1,1)) #ARIMA 1,1,1
results_arima_1_1 = model_arima_1_1.fit()
print(results_arima_1_1.summary())

#------------------- Find MA 2
model_arima_2_1 = ARIMA(df.market_value, order=(2,1,1)) #ARMA 2,1,1
results_arima_2_1 = model_arima_2_1.fit()
print(results_arima_2_1.summary())
#In Summary P > |z| should be lower, the lower the better

#Define a Log Likelihood Ratio (LLR) test to compare the models
def LLR_test(mod_1, mod_2, DF=1):
    L1 = mod_1.fit().llf
    L2 = mod_2.fit().llf
    LR = (2*(L2-L1))
    p = chi2.sf(LR, DF).round(3)
    return p

print("LLR Test:",LLR_test(model_arima_1_1, model_arima_2_1)) #0.0 is consodered high value


#------------ Print resisulta
print("Residuals:",results_arima_2_1.resid)

#---------------- AIC values
#-- For best model AIC should be least and LLR should be max
print("AIC model_arma_1_1:",results_arima_1_1.aic)
print("AIC model_arma_2_1:",results_arima_2_1.aic)

