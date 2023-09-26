import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

close = pd.read_csv('./data/sp500_close.csv', index_col=0, parse_dates=True)

#Get closing price for Goog stock
goog = close[['GOOG']].copy().dropna()
#goog.plot()
#plt.show()

# Take the log returns
# Add 1 as log(0) = undefined
goog_ret = np.log(goog.pct_change(1) + 1)
#goog_ret.plot()
#plt.show()

# Create a new column with rolling mean for past 10 values
goog['SMA-10'] = goog['GOOG'].rolling(10).mean()

# Plot the data
#goog.plot(figsize=(10, 5))
#plt.show()

# Now work on a multi-variate series
# Get closing price of 2 stocks
goog_aapl = close[['GOOG', 'AAPL']].copy().dropna()

# by hand calculation of covariance of 2 data points
# data starts from 3-27-2014
# Value of two stocks on 2 days
# Date     GOOG   AAPL
#2014-03-27  558.46  76.7799
#2014-03-28  559.99  76.6942
# Mean      559.225  76.738
# Variance for GOOG: (559.225 - 558.46)^2 + (559.225 - 559.99)^2/1 -> Divide by N-1, here N = 2
# Variance for GOOG: (.765)^2 + (-.765)^2/1 = (.585 + .585)/1 = 1.17 
# Variance for AAPL: (76.738 - 76.779)^2 + (76.738 - 76.6942)^2/1
# Variance for AAPL: (-.041)^2 + (.0438)^2/1 = (.0017 + .0019)/1 = .0036
# Covariance between GOOG-AAPL = sum((x-Xmean).(y-Ymean))
# = sum((559.225 - 558.46).(76.738 - 76.779) + (559.225 - 559.99).(76.738 - 76.6942))/1)
# = sum((0.765).(-.041) + (-.765).(.0438) = (-.03136 - .0335) = -.065
# Covariance Matrix(X,Y) = [var(x) cov(x,y) ]
#                          [cov(x,y)  var(y)]
#                        = [1.17   -.065]
#                          [-.065  .0036] 
# Cov() return covriance matrix, note this value will change based
# on rolling window size
print("Head values",goog_aapl.head())
cov = goog_aapl.rolling(2).cov()
print("Covariance",cov.head(6))

# by hand calculation of correlation of 2 data points
# corr (x,y) = cov(x,y)/sd(x).sd(y) (Pearson Correlation)
# sd(x) = sqrt(var(x))
# from above calculations of cov
# cov (GOOG,AAPL) = -0.065
# sd(GOOG) = sqrt(1.17) = 1.081
# sd(AAPL) = sqrt(.0036) = .06
# corr(GOOG,AAPL) = -.065 / (1.081 * .06)=-.065/.0648 = -1
# corr(GOOG,AAPL) matrix =          GOOG AAPL
#                            GOOG   [1.0   -1.0]
#                            AAPL   [-1.0   1.0 ]
#  Correlation with itself is always 1.0 
corr = goog_aapl.rolling(2).corr()
print("Correlation",corr.head(6))

# Compute log return of multivarite series
goog_aapl_ret = np.log(1 + goog_aapl.pct_change(1))

goog_aapl_ret['GOOG-SMA-50'] = goog_aapl_ret['GOOG'].rolling(50).mean()
goog_aapl_ret['AAPL-SMA-50'] = goog_aapl_ret['AAPL'].rolling(50).mean()

goog_aapl_ret.plot()
#plt.show()

#Find the rolling correlation
#Compute the correltion of columns
corr = goog_aapl_ret[['GOOG', 'AAPL']].rolling(50).corr()
print("Correlation Tail with rolling 50:",corr.tail(6))
