import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import boxcox

df = pd.read_csv('./data/airline_passengers.csv', index_col='Month', parse_dates=True)

df['Passengers'].plot(figsize=(10, 4))
plt.title("Raw Data")
plt.show()

df['SqrtPassengers'] = np.sqrt(df['Passengers'])
df['SqrtPassengers'].plot(figsize=(10, 4))
plt.title("SQRT Data")
plt.show()

df['LogPassengers'] = np.log(df['Passengers'])
df['LogPassengers'].plot(figsize=(10, 4))
plt.title("Log Data")
plt.show()

data, lam = boxcox(df['Passengers']) #lam is the lamda param used in the boxcox transform
df['BoxCoxPassengers'] = data
df['BoxCoxPassengers'].plot(figsize=(10, 4))
plt.title("Box Cox Data")
plt.show()

#-------------------- Plot Histograms
df['Passengers'].hist(bins=20)
plt.title("Passenger Histogram")
plt.show()
df['SqrtPassengers'].hist(bins=20)
plt.title("SqrtPassenger Histogram")
plt.show()
df['LogPassengers'].hist(bins=20)
plt.title("LogPassenger Histogram")
plt.show()
df['BoxCoxPassengers'].hist(bins=20)
plt.title("BoxCoxPassenger Histogram")
plt.show()








