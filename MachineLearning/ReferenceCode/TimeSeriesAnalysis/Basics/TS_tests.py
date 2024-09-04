import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import statsmodels.graphics.tsaplots as sgt #ACF/PACF plots
import statsmodels.tsa.stattools as sts 
from statsmodels.tsa.seasonal import seasonal_decompose
import seaborn as sns

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


# Show the Quantile-Quantile (QQ) Plot
import scipy.stats
import pylab
scipy.stats.probplot(df_copy.market_value, plot = pylab)
pylab.show()

#Dicker-Fuller Test
#Null Hypothesis : Non stationarity

# (-1.7369847452352445, 0.4121645696770618, 18, 5002, {'1%': -3.431658008603046, '5%': -2.862117998412982, '10%': -2.567077669247375}, 39904.880607487445)
# -1.7369847452352445: The test statistic
# 0.4121645696770618 : MacKinnon’s approximate p-value based on MacKinnon:
# pvalue% is the change of not rejecting the number
# 41% chance of not rejecting the null.
# lower pvalue signified stationary data. 
# 18 : Used lags 
adf = sts.adfuller(df.market_value)
print("DF Test",adf)

#p-value Null hypothesis
#Null hypothesis: 2 samples are same
#if p-value is high, null hypothesis is true
#if p-value is low, null hypothesis is not true and can be rejected

#General Null Hypothesis: There is no difference between things


#Seasonality plot - Additive
s_dec_multiplicative = seasonal_decompose(df.market_value, model = "additive")
s_dec_multiplicative.plot()
plt.title("Seasonality using Additive Mode",size=14)
plt.show()

#Seasonality plot - Multiplicative
s_dec_multiplicative = seasonal_decompose(df.market_value, model = "multiplicative")
s_dec_multiplicative.plot()
plt.title("Seasonality using Multiplicative Mode",size=14)
plt.show()

#Auto Correlation Function (ACF) - Use with MA
sgt.plot_acf(df.market_value, lags = 40, zero = False) #Alwats define lags, and zero
plt.title("ACF S&P", size = 20)
plt.show()

#Partial Auto Correlation Function (PACF) - Use with AR
sgt.plot_pacf(df.market_value, lags = 40, zero = False, method = ('ols')) #Default values
plt.title("PACF S&P", size = 20)
plt.show()

#Information Criterion (AIC/BIC)