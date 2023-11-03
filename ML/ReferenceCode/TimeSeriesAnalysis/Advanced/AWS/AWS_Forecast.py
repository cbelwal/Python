import pandas as pd
from AWSConfig import AWSConfig

dfOrig = pd.read_csv('./data/SPY.csv', index_col='Date', parse_dates=True)
data = dfOrig.loc['2018-01-10':'2021-01-11'] 

# add in missing dates
date_range = pd.date_range(data.index[0], data.index[-1])
date_range

# make a new dataframe containing all dates between the given ranges
df = pd.DataFrame(index=date_range)
df.head()

# Now merge new dataframe with all dates, with orig. data which is missing some dates
# inner join is intersection, and outer join is union
df = df.join(data, how='outer')
df.head()

# fill in missing data
# ffill option: Fill NA/NaN values by propagating the last valid observation to next
df[['Open', 'High', 'Low', 'Close', 'Adj Close']] = \
    df[['Open', 'High', 'Low', 'Close', 'Adj Close']].fillna(method='ffill') # propagate last value for NA
df['Volume'] = df['Volume'].fillna(0) # Fill NAs with 0

# AWS Forecast requires a column called item_id
df['item_id'] = 'SPY'

# leave the last 30 points for forecast comparison
FORECAST_LENGTH = 30
train = df.iloc[:-FORECAST_LENGTH]

# AWS differentiates between "target time series" and "related time series"
train_target_series = train[['Close', 'item_id']]
train_related_series = train[['Open', 'High', 'Low', 'Volume', 'item_id']]

# Save the data which we will upload to S3 later
train_target_series.to_csv("./data/daily_price_target_series.csv", header=None)
train_related_series.to_csv("./data/daily_price_related_series.csv", header=None)

# Upload files to CSV
aws = AWSConfig()
aws.uploadToS3("TargetSeries","./data/daily_price_target_series.csv")
