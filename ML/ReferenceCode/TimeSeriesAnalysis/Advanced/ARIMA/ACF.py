# ACF is used for the MA (Moving Average) component
# At lag k, this is the correlation between 
# series values that are k intervals apart. 

from statsmodels.graphics.tsaplots import plot_pacf, plot_acf

import numpy as np
import matplotlib.pyplot as plt


def show_acf(x,title=""):
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_acf(x, ax=ax)
    plt.title("ACF:" + title)
    plt.show()


# 1. Generate total random data
# iid (Independent and identically distributed) noise - 
x0 = np.random.randn(1000)
show_acf(x0,"Random - IID noise") #No relation with past values will show


# 2. Generate data where value at index i is based on value at index i-1 
# to some degree with a +ve relation ship

# Note that unlike PACF which is dependent on last value for AR purposes,
# for MA only actual error terms are considered. And the error at time t is not
# dependent on error at time t-1 
errors = 0.1 * np.random.randn(1000) #Compute errors as a Gaussian distribution
ma1 = []
for i in range(1000):
  if i >= 1:
    # Not dependent on previous term ma1[i-1]
    # but dependent on previous error term at i-1 only
    x = 0.5 * errors[i-1] + errors[i] 
  else:
    x = errors[i]
  ma1.append(x)
ma1 = np.array(ma1)
show_acf(ma1,"0.5 on last error term")

# 3. -------------------------------

errors = 0.1 * np.random.randn(1000)
ma2 = []
for i in range(1000):
  x = 0.5 * errors[i-1] - 0.3 * errors[i-2] + errors[i]
  ma2.append(x) # Add new value
ma2 = np.array(ma2)
show_acf(ma2,"-0.3 on error term i-2, 0.5 on error term i-1")

# 4. -------------------------------
errors = 0.1 * np.random.randn(1000)
ma3 = []
for i in range(1000):
  x = 0.5 * errors[i-1] - 0.3 * errors[i-2] + 0.7 * errors[i-3] + errors[i]
  ma3.append(x)
ma3 = np.array(ma3)
show_acf(ma3,"0.5 on i-1,-0.3 on i-2,0.7 on i-3")

# 5. -------------------------------
errors = 0.1 * np.random.randn(1000)
ma4 = []
for i in range(1000):
  x = 0.5 * errors[i-1] - 0.3 * errors[i-2] + 0.7 * errors[i-3] + \
      0.2 * errors[i-4] - 0.8 * errors[i-5] - 0.9 * errors[i-6] + errors[i] 
  ma4.append(x)
ma4 = np.array(ma4)
show_acf(ma4,"i-1 to i-6, both +ve and -ve")