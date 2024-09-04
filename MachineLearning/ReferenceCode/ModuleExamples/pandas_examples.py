import pandas as pd
import numpy as np

#Create dataframe from list
data=[3,6,2,9,1]
df = pd.DataFrame(data, columns=['Numbers'])
print("df:",df)

newDf = df.iloc[-1:] 
print("newDf with only last row",newDf)

# Create datafrmae from random numbers
# Generate numbers from a normal distriction of dimenstions 10x4
rndNormal = np.random.randn(10, 4)
print("Normal numbers:",rndNormal)
df = pd.DataFrame(rndNormal, columns=['col1','col2','col3','col4'])
print("Dataframe:",df)
