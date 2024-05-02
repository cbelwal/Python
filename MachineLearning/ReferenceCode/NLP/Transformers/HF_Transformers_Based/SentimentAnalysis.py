# Sentiment analysis using HF library
# Built with ref. to the Transformers course in Udemy offered by TheLazyProgrammer.

from transformers import pipeline

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import time

from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import torch # for GPU/CUDA check

from sklearn import metrics

def plotConfusionMatrix(cm, title="Confusion Matrix"):
    #Plot confusion Matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
      
    fig, ax = plt.subplots(figsize=(10,5))
    disp.plot(ax=ax)
    plt.suptitle(title,fontweight='bold')
    plt.show()


classifier = pipeline("sentiment-analysis")

print("GPU available:",torch.cuda.is_available())

# Sample calls to the classifier model
print(classifier("What a great movie!"))
print(classifier("This movie was not bad at all!"))
print(classifier("Not a good movie at all!"))

# Now read the CSV file
df_ = pd.read_csv('./data/AirlineTweets.csv')
print(df_.head())

# Make a new df with only 2 columns, text and its classification
df = df_[['airline_sentiment', 'text']].copy()

# Remove all rows with Neutral classification as 
# classification only support postive and negative categories 
df = df[df.airline_sentiment != 'neutral'].copy()
target_map = {'positive': 1, 'negative': 0}
df['target'] = df['airline_sentiment'].map(target_map)

df = df.head(100) #Only do 100 rows since we do not have GPU
# Look at the df again
print(df.head())

# convert df of text to a list
texts = df['text'].tolist()

print("Starting classification ...")
start = time.time()
predictions = classifier(texts)
mins = (time.time() - start)/60.0
print("Calculation took %s mins " % mins)
#print(predictions)

# Compute scores
preds = [1 if d['label'].startswith('P') else 0 for d in predictions]
preds = np.array(preds)

cm = confusion_matrix(df['target'], preds, normalize='true')
print("Confusion Matrix:",cm)
plotConfusionMatrix(cm)

# Other metrics
# NOTE: f1 score will depend on which label is assigned as 0 and which
# one is assigned as 1. Hence we will compute F1 for both classes

print("F1 Score:",f1_score(df['target'],preds))
print("Reversed F1 Score:",f1_score(1 - df['target'],1 - preds))

# AUC does not depend on the classification and this can be demonstrated by following
# Both values will be the same
print("roc_auc_score:",roc_auc_score(df['target'],preds))
print("Reversed roc_auc_score:",roc_auc_score(1 - df['target'],1 - preds))
