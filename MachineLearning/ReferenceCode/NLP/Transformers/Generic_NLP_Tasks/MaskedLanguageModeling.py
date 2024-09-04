# pip install transfomers

# Masked Language Modeling (MLM) - also used in Article spinning

import numpy as np
import pandas as pd

from transformers import pipeline

df = pd.read_csv('./data/bbc_text_cls.csv') # Read list of new articles
labels = set(df['labels'])
# only select business articles and column having 'text'
texts = df[df['labels'] == 'business']['text'] 

# Randomly select text
np.random.seed(100)
# will select random index between 0 and texts.shape[0] (which has all rows) 
i = np.random.choice(texts.shape[0])
doc = texts.iloc[i]

print("Number of randomly selected documents:",len(doc))

# define the pipeline for the masked language model
mlm = pipeline('fill-mask') # distilroberta-base is selected

# Try with a sample
print("Samples:",mlm('The cat <mask> over the fence'))

print("Samples:",mlm('The <mask> jump over the fence'))

# Another test with an article in the BBC article list
text = 'Shares in <mask> and plane-making ' + \
  'giant Bombardier have fallen to a 10-year low following the departure ' + \
  'of its chief executive and two members of the board.'

mlm(text)

