import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from keras.layers import Dense, Input
from keras.models import Model

#Dataset from: https://www.kaggle.com/shivamkushwaha/bbc-full-text-document-classification
input_file= 'c:/users/chaitanya belwal/.datasets/nlp/bbc_text_cls.csv'


df = pd.read_csv(input_file)
# map classes to integers from 0...K-1
df['targets'] = df['labels'].astype("category").cat.codes

print(df['targets'])


df_train, df_test = train_test_split(df, test_size=0.3)

tfidf = TfidfVectorizer(stop_words='english')
Xtrain = tfidf.fit_transform(df_train['text'])
Xtest = tfidf.transform(df_test['text'])

Ytrain = df_train['targets']
Ytest = df_test['targets']

#Number of classes, add 1 since we are base 0
K = df['targets'].max() + 1
# input dimensions
D = Xtrain.shape[1]

print("Total Dimensions:",D) # D will be large, for each work


# build model
i = Input(shape=(D,))
x = Dense(300, activation='relu')(i) #Number of neurons at 300 is arbitary
x = Dense(K)(x) # softmax included in loss, last laters

model = Model(i, x)

print("Model Summary", model.summary())
#Total input neurons =: 25,200 (D)
#Total params in Dense: 25,200 * 300 + 300 (for intercept) = 7,560,300
#Total params in Dense: 300 * 5 + 5 (for intercept) = 1505

model.compile(
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  optimizer='adam',
  metrics=['accuracy']
)

# data must not be sparse matrix before passing into tensorflow
Xtrain = Xtrain.toarray()
Xtest = Xtest.toarray()

run = model.fit(
  Xtrain, Ytrain,
  validation_data=(Xtest, Ytest),
  epochs=7,
  batch_size=128,
)

# Plot loss per iteration
plt.plot(run.history['loss'], label='train loss')
plt.plot(run.history['val_loss'], label='val loss')
plt.legend()
plt.show()

# Plot accuracy per iteration
plt.plot(run.history['accuracy'], label='train acc')
plt.plot(run.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()

#Show histogram
df['labels'].hist()
plt.show()

#TODO: Pass Predictions