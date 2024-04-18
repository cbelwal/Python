import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

df = pd.read_csv('./data/airline_passengers.csv', index_col='Month', parse_dates=True)
df.index.freq = 'MS'

result = adfuller(df['Passengers'])

print(result)

def adf_components(x,title=""):
  res = adfuller(x)
  print("---------- ADF Test for:", title)
  print("Test-Statistic:", res[0])
  print("P-Value:", res[1])
  if res[1] < 0.05:
    print("Stationary") # Alternate Hypothesis Holds
  else:
    print("Non-Stationary") #Null Hypothesis Holds

# Do component test of ADF for main distribution
adf_components(df['Passengers'])

# Check ADF for a strong stationarity signal: from normal distribution
adf_components(np.random.randn(100),"Normal")

# Check ADF for a strong stationarity signal: from gamma distribution
adf_components(np.random.gamma(1, 1, 100),"Gamma")

df['LogPassengers'] = np.log(df['Passengers'])

# Check ADF for log values
adf_components(df['LogPassengers'],"Log")


df['Diff'] = df['Passengers'].diff()
# ADF for 1st Diff - This can be close to stationary as this is what the d value in ARIMA does
adf_components(df['LogPassengers'].dropna(),"1st Diff")

df['DiffLog'] = df['LogPassengers'].diff()
# ADF for 1st Diff of Log 
adf_components(df['DiffLog'].dropna(),"Log 1st Diff")








