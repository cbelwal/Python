import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.holtwinters import Holt

df = pd.read_csv('./data/airline_passengers.csv', index_col='Month', parse_dates=True)

holt = Holt(
    df['Passengers'],
    initialization_method='legacy-heuristic')

res_h = holt.fit()
df['Holt'] = res_h.fittedvalues
df[['Passengers', 'Holt']].plot()
plt.show()

#----------------- Now split on training and test
N_test = 15
train = df.iloc[:-N_test]
test = df.iloc[-N_test:]

train_idx = df.index <= train.index[-1]
test_idx = df.index > train.index[-1]

holt = Holt(
    train['Passengers'],
    initialization_method='legacy-heuristic')

res_h = holt.fit()

df.loc[train_idx, 'Holt_split'] = res_h.fittedvalues #Give the 
df.loc[test_idx, 'Holt_split'] = res_h.forecast(N_test)
df[['Passengers', 'Holt_split']].plot()
plt.title("With Train/Test Split")
plt.show()
