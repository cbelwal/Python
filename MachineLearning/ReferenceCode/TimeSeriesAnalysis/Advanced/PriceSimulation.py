import numpy as np
import matplotlib.pyplot as plt

# number of time steps
T = 1000
# initial price
P0 = 10
# drift -> Contols the trend of the time series
mu = 0.001

# Set arrays
# last log price
last_p = np.log(P0)
log_returns = np.zeros(T)
prices = np.zeros(T)
for t in range(T):
  # sample a log return
  # By multiplying by mu(=.01) we get a random number 
  # from normal distribution with mean .01
  r = 0.01 * np.random.randn()

  # compute the new log price
  p = last_p + mu + r

  # store the return and price
  log_returns[t] = r + mu #We can so summation here as this is a log operation
  # p = log return of prices(P) => loge P = p => P = e^p
  # => P = np.exp(p)
  prices[t] = np.exp(p) 

  # assign last_p
  last_p = p

plt.figure(figsize=(10, 4))
plt.plot(prices)
plt.title("Simulated Prices of a stock")
plt.show()
