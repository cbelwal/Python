import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Load Data
df_raw = pd.read_csv("./Data/index2018.csv")

#Create a copy of data
df_copy = df_raw.copy()

#Shows mean, stddev etc for data
print("Data details:",df_copy.describe()) 

#Show true/false if value is na
df_copy.isna() 

#show sum of isna in data
df_copy.isna().sum() 

#Plot data for a specific column (here name is index)
df_copy.spx.plot() 

#More attributes for plot function
df_copy.spx.plot(figsize=(10,5), title = "S&P500 Prices")
plt.show()



#Change a value to datetime datatype
df_copy.date = pd.to_datetime(df_copy.date, dayfirst = True)

#set index based on a specific column
#index cannot be changed
df_copy.set_index("date",inplace=True) 

#Set the desired freq of data
#d = day, h = hour, w = week, b= business day (m-f)
df_copy=df_copy.asfreq('d') 

#Fill missing(na) values using Front-filling
#Front fill will copy the data from the last period into current period
df_copy.spx=df_copy.spx.fillna(method='ffill')

#Fill missing(na) values using Back-filling
#Back fill will copy the data from the next period into current period
df_copy.ftse=df_copy.ftse.fillna(method='bfill')

#Fill missing(na) values using given values
#In this case the given valu is the avg of all other values
df_copy.dax=df_copy.dax.fillna(value = df_copy.dax.mean())

#Remove multiple columns from the DF
del df_copy['dax'], df_copy['ftse'],df_copy['nikkei']

#iloc - index location

