# In traditional ML problem we split on train and test
# and then do a k-fold validation to get different train and test test
# In time series data k-Fold validation cannot be done as we cant mix future 
# values with past values.
# Hence, for time series we do a Walk Forward Validation where we take consecutive 
# data in train and test (test follows train) and increase the size of train and test
# in each iteration.
#
# Just like with kFold validation, with Walk forward validation we can find the 
# best hyper-parameters. 
#
# The following shows how the walk-through validation is done
# * - is the training set, and # - if the test set
# Iteration 1: *************######
# Iteration 2: ***************#######
# Iteration 3: ******************########
# Iteration 4: *********************#########
# Iteration 5: ************************##########
# Iteration 6: **************************###########

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import itertools

from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing

df = pd.read_csv('./data/airline_passengers.csv', index_col='Month', parse_dates=True)
df.index.freq = 'MS'
# Assume the forecast horizon we care about is 12
# Validate over 10 steps
h = 12
steps = 10
Ntest = len(df) - h - steps + 1

# Configuration hyperparameters to try
# These are fed into a tuple where multiple combinations of them are automatically 
# tried

# These options are used in the ExponentialSmoothing Function 
trend_type_list = ['add', 'mul']
seasonal_type_list = ['add', 'mul']
damped_trend_list = [True, False]
init_method_list = ['estimated', 'heuristic', 'legacy-heuristic']
use_boxcox_list = [True, False, 0]

def walkforward(
    trend_type,
    seasonal_type,
    damped_trend,
    init_method,
    use_boxcox,
    debug=False):

  # store errors
  errors = []
  seen_last = False
  steps_completed = 0

  # Increase the training size, and shift test size by 1
  for end_of_train in range(Ntest, len(df) - h + 1):
    # We don't have to manually "add" the data to our dataset
    # Just index it at the right points - this is a "view" not a "copy"
    # So it doesn't take up any extra space or computation

    # Keep extending the train test, and then extend the test set too
    train = df.iloc[:end_of_train]
    # Test index is always h size when the train ends
    test = df.iloc[end_of_train:end_of_train + h]

    if test.index[-1] == df.index[-1]:
      seen_last = True
    
    steps_completed += 1

    hw = ExponentialSmoothing(
        train['Passengers'],
        initialization_method=init_method,
        trend=trend_type,
        damped_trend=damped_trend,
        seasonal=seasonal_type,
        seasonal_periods=12,
        use_boxcox=use_boxcox)
    res_hw = hw.fit()

    # compute error for the forecast horizon
    fcast = res_hw.forecast(h)
    error = mean_squared_error(test['Passengers'], fcast)
    errors.append(error)
  
  if debug:
    print("seen_last:", seen_last)
    print("steps completed:", steps_completed)

  return np.mean(errors)

# Single call
walkforward('add', 'add', False, 'legacy-heuristic', 0, debug=True)


# Create a tuple of options, as this will allow us to use itertools
# Iterate through all possible options (i.e. grid search)
tuple_of_option_lists = (
    trend_type_list,
    seasonal_type_list,
    damped_trend_list,
    init_method_list,
    use_boxcox_list,
)

best_score = float('inf')
best_options = None
# itertools.product will automatically create all permutations of the 
# supplied params
# * denotes a tuple, and ** is hash as per Python docs
for x in itertools.product(*tuple_of_option_lists):
  # Print all the permuations for verifications
  # print(x)
  score = walkforward(*x)

    # Print out the best combination of the permuations
  if score < best_score:
    print("Best score,options so far:", score,x)
    best_score = score
    best_options = x
