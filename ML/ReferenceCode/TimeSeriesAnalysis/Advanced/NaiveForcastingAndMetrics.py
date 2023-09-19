import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_percentage_error, \
  mean_absolute_error, r2_score, mean_squared_error

#, \
#    mean_absolute_error, r2_score, mean_squared_error

df = pd.read_csv('SPY.csv', index_col='Date', parse_dates=True)