import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.distributions import chi2
import statsmodels.tsa.stattools as sts 
from pmdarima.arima import auto_arima


#Load Data
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
#---------------------------------------

# Use ACF plot to analyze coefficients for MA

# The (p,d,q) order of the model for the autoregressive, differences, and moving
model_auto_1 = auto_arima(df_copy.market_value[1:])
print("Auto ARIMA Summary",model_auto_1.summary())
print("Auto ARIMA Parameters",model_auto_1)

model_auto_2 = auto_arima(df_copy.market_value[1:], exogenous = df_copy[['spx', 'dax', 'ftse']][1:], m = 5,
                       max_order = None, max_p = 7, max_q = 7, max_d = 2, max_P = 4, max_Q = 4, max_D = 2,
                       maxiter = 50, alpha = 0.05, n_jobs = -1, trend = 'ct', information_criterion = 'oob',
                       out_of_sample = int(len(df_copy)*0.2))

print("Auto ARIMA Parameters",model_auto_2)
# !!! Important Note: In pdmarima v1.5.2, out_of_sample_size is replaced with out_of_sample, so make sure to use the latter!
# exogenous -> outside factors (e.g other time series)
# m -> seasonal cycle length
# max_order -> maximum amount of variables to be used in the regression (p + q)
# max_p -> maximum AR components
# max_q -> maximum MA components
# max_d -> maximum Integrations
# maxiter -> maximum iterations we're giving the model to converge the coefficients (becomes harder as the order increases)
# alpha -> level of significance, default is 5%, which we should be using most of the time
# n_jobs -> how many models to fit at a time (-1 indicates "as many as possible")
# trend -> "ct" usually
# information_criterion -> 'aic', 'aicc', 'bic', 'hqic', 'oob' 
#        (Akaike Information Criterion, Corrected Akaike Information Criterion,
#        Bayesian Information Criterion, Hannan-Quinn Information Criterion, or
#        "out of bag"--for validation scoring--respectively)
# out_of_sample -> validates the model selection (pass the entire dataset, and set 20% to be the out_of_sample_size)


