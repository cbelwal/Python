import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.graphics.tsaplots as sgt
from scipy.stats.distributions import chi2
import statsmodels.tsa.stattools as sts 
from statsmodels.tsa.statespace.sarimax import SARIMAX

#Load Data
df_raw = pd.read_csv("./Data/index2018.csv")

#Data Sanitization--------------
df_copy = df_raw.copy()
df_copy.date = pd.to_datetime(df_copy.date, dayfirst = True)
df_copy.set_index("date", inplace=True)
df_copy=df_copy.asfreq('b')
df_copy=df_copy.fillna(method='ffill')
df_copy['market_value']=df_copy.spx

#del df_copy['spx'] Done delete this as its used for exogenous variable
del df_copy['dax']
del df_copy['ftse']
del df_copy['nikkei']
size = int(len(df_copy)*0.8)
df, df_test = df_copy.iloc[:size], df_copy.iloc[size:]
#---------------------------------------

# Use ACF plot to analyze coefficients for MA

#------------------- Find MA 1
# The (p,d,q) order of the model for the autoregressive, differences, and moving
model_sarimax_1_1 = SARIMAX(df.market_value, exog=df.spx, order=(1,1,1), ) #ARIMA 1,1,1, exog = MAX
results_sarimax_1_1 = model_sarimax_1_1.fit()
print("Summary:",results_sarimax_1_1.summary())

#------------ Print resisulta
print("Residuals:",results_sarimax_1_1.resid)

#---------------- AIC values
#-- For best model AIC should be least and LLR should be max
print("AIC model_arma_1_1:",results_sarimax_1_1.aic)
