# pip install transfomers

# Zero shot classification allow the model to classify
# the input text into given categories without apriori knowledge
# The choices for classification have to be explicitly stated
from transformers import pipeline

import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

# Set device=0 when GPU is available
classifier = pipeline("zero-shot-classification") #, device=0)

print("Sample run #1:", classifier("This is a great movie", candidate_labels=["positive", "negative"]))

# Let us now extract text from bbc data set
df = pd.read_csv('./data/bbc_text_cls.csv')

# Note the above df contains more columns but the columns of interest are 
# only 'text' and 'labels'
# Extract the labels
labels = list(set(df['labels']))
print("All possible labels of text:",labels)


# pick some sample text from row 801
srow = 801
sample_text = df.iloc[srow]['text']
actual_class = df.iloc[srow]['labels']
print("Classification from BBC corpus #1:",classifier(df.iloc[srow]['text'], candidate_labels=labels))
print("Actual class:",actual_class)