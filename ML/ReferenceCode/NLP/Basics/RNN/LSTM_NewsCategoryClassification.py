import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import LSTM, GRU, SimpleRNN, Embedding
from keras.models import Model
from keras.losses import SparseCategoricalCrossentropy

#Dataset from: https://www.kaggle.com/shivamkushwaha/bbc-full-text-document-classification
input_file= 'c:/users/chaitanya belwal/.datasets/nlp/bbc_text_cls.csv'


df = pd.read_csv(input_file)
# map classes to integers from 0...K-1
df['targets'] = df['labels'].astype("category").cat.codes 

#2224 Documents and Targets
print(df['targets'])

df_train, df_test = train_test_split(df, test_size=0.3)

#1557 in Train
#667 in train

# Convert sentences to sequences
MAX_VOCAB_SIZE = 2000
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
tokenizer.fit_on_texts(df_train['text'])

Xtrain = tokenizer.texts_to_sequences(df_train['text']) #1557 documents of differerent sequence length
Xtest = tokenizer.texts_to_sequences(df_test['text'])   #668 documents

Ytrain = df_train['targets']
Ytest = df_test['targets']

#Number of classes, add 1 since we are base 0
K = df['targets'].max() + 1

# get word -> integer mapping
word2idx = tokenizer.word_index
V = len(word2idx)
print('Found %s unique tokens.' % V)

# pad sequences so that we get a N x T matrix
data_train = pad_sequences(Xtrain)
print('Shape of data train tensor:', data_train.shape)

# get sequence length
T = data_train.shape[1] #3484 is the number of columns
# Create the model

data_test = pad_sequences(Xtest, maxlen=T)
print('Shape of data test tensor:', data_test.shape)

# We get to choose embedding dimensionality
D = 20

# Note: we actually want to the size of the embedding to (V + 1) x D,
# because the first index starts from 1 and not 0.
# Thus, if the final index of the embedding matrix is V,
# then it actually must have size V + 1.

i = Input(shape=(T,)) #3484 columns
x = Embedding(V + 1, D)(i) #Embedding layers creates the word vector, depth of embedding is D
#32 is the number of cells
x = LSTM(32, return_sequences=True)(x) #This is the main change compared to the one using CNN
x = GlobalMaxPooling1D()(x)
x = Dense(K)(x)

model = Model(i, x)

# Compile and fit
model.compile(
  loss=SparseCategoricalCrossentropy(from_logits=True),
  optimizer='adam',
  metrics=['accuracy']
)

print('Training model...')
r = model.fit(
  data_train,
  df_train['targets'],
  epochs=2, #Very low number to complete faster
  validation_data=(data_test, df_test['targets'])
)

# Plot loss per iteration
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()

# Plot loss per iteration
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()

#predictions = model.predict(X)