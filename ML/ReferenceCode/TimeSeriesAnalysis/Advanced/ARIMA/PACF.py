# PACF is used for the AR (Auto Regressive) component
# Partial autocorrelation function (PACF). At lag k, this is the correlation 
# between series values that are k intervals apart, 
# and accounting for the values of the intervals between.

from statsmodels.graphics.tsaplots import plot_pacf, plot_acf

import numpy as np
import matplotlib.pyplot as plt


def show_pacf(x,title=""):
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_pacf(x, ax=ax)
    plt.title("PACF:" + title)
    plt.show()


# 1. Generate total random data
x0 = np.random.randn(1000)
show_pacf(x0,"Random") #No relation with past values will show


# 2. Generate data where value at index i is based on value at index i-1 
# to some degree with a +ve relation ship
x1 = [0]
for i in range(1000):
  # value at index i is atleast .5 of the value at index i - 1, and these is a random value added   
  x = 0.5 * x1[-1] + 0.1 * np.random.randn() # randn can generate any real number with numbers close to 0 more likely
  x1.append(x)
x1 = np.array(x1) # Convert to np.array for plotting purposes
show_pacf(x1,"0.5 on last value")

# 3. Generate data where value at index i is based on value at index i-1 
# to some degree with a -ve relation ship
x1 = [0]
for i in range(1000):
  # value at index i is atleast .5 of the value at index i - 1, and these is a random value added   
  x = -0.5 * x1[-1] + 0.1 * np.random.randn() # randn can generate any real number with numbers close to 0 more likely
  x1.append(x)
x1 = np.array(x1) # Convert to np.array for plotting purposes
show_pacf(x1,"-0.5 on last value")


# 4. Generate data where values at index i is based on values at index i-1, and i-2 
# to some degree with a +ve and -ve relation ship
x2 = [0, 0]
for i in range(1000):
  x = 0.5 * x2[-1] - 0.3 * x2[-2] + 0.1 * np.random.randn()
  x2.append(x)
x2 = np.array(x2)
show_pacf(x2,"0.5 on i-1,-0.3 on i-2")

# 5. Generate data where values at index i is based on values at index i-1, i-2,i-3 
# to some degree with a +ve and -ve relation ship
x5 = [0, 0, 0, 0, 0]
for i in range(1000):
  x = 0.5 * x5[-1] - 0.3 * x5[-2] - 0.6 * x5[-5] + 0.1 * np.random.randn()
  x5.append(x)
x5 = np.array(x5)
show_pacf(x5,"0.5 on i-1,-0.3 on i-2,-0.6 on i-5")
