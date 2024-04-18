import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

from datetime import datetime
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.api import VAR
from sklearn.metrics import r2_score
# StandardScaler removes the mean and scales each feature/variable 
# to unit variance. This operation is performed feature-wise in an 
# independent way. StandardScaler can be influenced by outliers 
# (if they exist in the dataset) since it involves the estimation 
# of the empirical mean and standard deviation of each feature.
from sklearn.preprocessing import StandardScaler

# Load multi-variate, this CSV file has temperature data on multiple cities
# However, data is not in an easy to consume format so we have have to do some 
# data fudging
#  
df = pd.read_csv('./data/temperature.csv')

# Time and date is split in multiple columns
# So combine into a string and then convert to property datetime 
# object
def parse_date(row):
  s = f"{row['year']}-{row['month']}-{row['day']}"
  return datetime.strptime(s, "%Y-%m-%d")

# Create date column using parse_date function
df['Date'] = df.apply(parse_date, axis=1)

# Create new df with rows having date for 2 cities
# The multi-variate series will contain temperature from only
# these 2 cities
auckland = df[df['City'] == 'Auckland'].copy().dropna()
stockholm = df[df['City'] == 'Stockholm'].copy().dropna()

# Keep on the temperatue column from the above DFs
auckland = auckland[['Date', 'AverageTemperatureFahr']].copy()
stockholm = stockholm[['Date', 'AverageTemperatureFahr']].copy()

# Set the index to date column, this is important is data is 
# to be treated as time series
auckland.set_index(auckland['Date'], inplace=True)
auckland.drop('Date', axis=1, inplace=True) # Remove date as its not needed any more
auckland.columns = ['AucklandTemp'] # Put a name to the column

#--- Repeat for Stockholm
# Set the index to date column, this is important is data is 
# to be treated as time series
stockholm.set_index(stockholm['Date'], inplace=True)
stockholm.drop('Date', axis=1, inplace=True) # Remove date as its not needed any more
stockholm.columns = ['StockholmTemp'] # Put a name to the column

print(stockholm.head())
print("Shape",stockholm.shape)

# join the two df with outer join
# outer join will be a union and there will be a row for each date
# even if there are missing values in one of the columns
joined = auckland.join(stockholm, how='outer') 
print("Joined Shape",joined.shape)

# Take the last 500 rows, as orig. 3000 rows is a very large number
joined_part = joined.iloc[-500:].copy()
joined_part.index.freq = 'MS'

# Show number of missing values
print("Missing Values:\n",joined_part.isna().sum())

# Fill in missing value via interpolate
# default method is 'linear'
# 'linear' will build a line between two (x1,y1) and (x2,y2)
# and fill in the missing values. 
# From df documentation: ‘linear’: Ignore the index and treat the values as equally spaced. This is the only method supported on MultiIndexes.assumes values are r
# Also See: https://www.johndcook.com/interpolator.html
joined_part.interpolate(inplace=True)


# Show number of missing values
print("Missing Values after interpolation:\n",joined_part.isna().sum())


# Split into train / test
Ntest = 12 # 12 mos worth of data
train = joined_part.iloc[:-Ntest].copy() # all except last 12 for train
test = joined_part.iloc[-Ntest:].copy() # last 12 for test

# Now scale the values. Though the temperature unit is same (F)
# they are significantly diverging to need to bring them to same scale
# IMPORTANT: Scaling has to be done after the train/test split, as train
# data needs to be run through fit_transform() while test data only needs
# transform(). The means and std used for scaling is the one used for train
# data
# 
# fit(): Compute the mean and std to be used for later scaling.
# fit_transform(): Fit to data, then transform it.
# transform(): Perform standardization by centering and scaling.

scaler_auckland = StandardScaler()
train['ScaledAuckland'] = scaler_auckland.fit_transform(
    train[['AucklandTemp']])
# test will use same scaling fit params as for training
test['ScaledAuckland'] = scaler_auckland.transform(test[['AucklandTemp']])

scaler_stockholm = StandardScaler()
train['ScaledStockholm'] = scaler_auckland.fit_transform(
    train[['StockholmTemp']])
# test will use same scaling fit params as for training
test['ScaledStockholm'] = scaler_auckland.transform(test[['StockholmTemp']])

# Now put scaled values in original df
train_idx = joined_part.index <= train.index[-1]
test_idx = joined_part.index > train.index[-1]

# Start from year 1974
joined_part.loc[train_idx, 'ScaledAuckland'] = train['ScaledAuckland']
joined_part.loc[test_idx, 'ScaledAuckland'] = test['ScaledAuckland']
joined_part.loc[train_idx, 'ScaledStockholm'] = train['ScaledStockholm']
joined_part.loc[test_idx, 'ScaledStockholm'] = test['ScaledStockholm']

cols = ['ScaledAuckland', 'ScaledStockholm']
#train[cols].plot(figsize=(10, 5))
#plt.show()

#plot_acf(train['ScaledAuckland'])
#plt.show()


#------------------- Do VAR calculation, Vector with AR but no MA components
t0 = datetime.now()
# train[cols] is the multi-variate time series
# cols naes is the scaled values
# VAR trains faster than VARMAX
model = VAR(train[cols]) #Only pass ScaledAuckland and ScaledStockholm
lag_order_results = model.select_order(maxlags=15)

print(lag_order_results.selected_orders)

results = model.fit(maxlags=15, ic='aic')
lag_order = results.k_ar #Compute how many ar params were used

# Find values used in training the model. This is computed based 
# on the lag order. Do not include the lag order in the computations
prior = train.iloc[-lag_order:][cols].to_numpy() # Need to convert to num_py array
#print(prior)

# forecast function still has to be called even on predicted values. 
# will forcast for 12 mos, there is not particular reason to choose 12
fcast = results.forecast(prior, Ntest)

print("Fcast",fcast)
joined_part.loc[train_idx, 'Train Pred Auckland'] = \
  results.fittedvalues['ScaledAuckland']
joined_part.loc[test_idx, 'Test Pred Auckland'] = \
  fcast[:,0]
joined_part.loc[train_idx, 'Train Pred Stockholm'] = \
  results.fittedvalues['ScaledStockholm']
joined_part.loc[test_idx, 'Test Pred Stockholm'] = \
  fcast[:,1]


#----------- Plots - Auckland
plot_cols = ['ScaledAuckland', 'Train Pred Auckland', 'Test Pred Auckland']
#Show all data points
joined_part.iloc[:][plot_cols].plot(figsize=(10, 5))
plt.title("Auckland - All Data points")
plt.show()
# Show only 100 data points
joined_part.iloc[-100:][plot_cols].plot(figsize=(10, 5))
plt.title("Auckland - Show 100 data points")
plt.show()

#----------- Plots - Stockholm
plot_cols = ['ScaledStockholm', 'Train Pred Stockholm', 'Test Pred Stockholm']
#Show all data points
joined_part.iloc[:][plot_cols].plot(figsize=(10, 5))
plt.title("Stockholm - All Data points")
plt.show()
# Show only 100 data points
joined_part.iloc[-100:][plot_cols].plot(figsize=(10, 5))
plt.title("Stockholm - Show 100 data points")
plt.show()

# Show R^2 values
# Auckland
y_pred = joined_part.loc[train_idx, 'Train Pred Auckland']
y_pred = y_pred.iloc[lag_order:] #Only consider values after lagorder
y_true = joined_part.loc[train_idx, 'ScaledAuckland']
y_true = y_true.iloc[lag_order:] #Only consider values after lagorder
print("Auckland Train R^2:", r2_score(y_true, y_pred))

y_pred = joined_part.loc[test_idx, 'Test Pred Auckland']
y_true = joined_part.loc[test_idx, 'ScaledAuckland']
print("Auckland Test R^2:", r2_score(y_true, y_pred))

# Stockholm
y_pred = joined_part.loc[train_idx, 'Train Pred Stockholm']
y_pred = y_pred.iloc[lag_order:]
y_true = joined_part.loc[train_idx, 'ScaledStockholm']
y_true = y_true.iloc[lag_order:]

print("Stockholm Train R^2:", r2_score(y_true, y_pred))

y_pred = joined_part.loc[test_idx, 'Test Pred Stockholm']
y_true = joined_part.loc[test_idx, 'ScaledStockholm']
print("Stockholm Test R^2:", r2_score(y_true, y_pred))