# Exponentially weighted moving average (EWMA)
# EWMA = a.x(t) + (1-a).ewma(t-1), x = value at time t 
# a = alpha, the decay factor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('./data/airline_passengers.csv', index_col='Month', parse_dates=True)

alpha = 0.2

# Compute pandas based EWMA
# adjust = False is important as with adjust=True, some other calculation is done
df['EWMA'] = df['Passengers'].ewm(alpha=alpha, adjust=False).mean()

df.plot() # Show the values of plot
plt.title("EWMA computed by Pandas")

manual_ewma = []
for x in df['Passengers'].to_numpy():
  if len(manual_ewma) > 0:
    # This equation can be derived, when current average is a function of last average
    xhat = alpha * x + (1 - alpha) * manual_ewma[-1]
  else:
    xhat = x #First values
  manual_ewma.append(xhat)
df['Manual'] = manual_ewma

df.plot() # Show the values of plot
plt.title("EWMA computed manually")
# Manual and Pandas lines overlap, showing they have same values
plt.show()
