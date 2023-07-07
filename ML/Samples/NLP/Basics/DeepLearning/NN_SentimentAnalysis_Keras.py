from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

import pandas as pd

import tensorflow as tf
import matplotlib.pyplot as plt

from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy


#Dataset from: https://www.kaggle.com/crowdflower/twitter-airline-sentiment

#--------------- Basic execution settings
useTestDataFile = False
useTFIDFVectorizer = True
#----------------------------------

#tweet_id,airline_sentiment,airline_sentiment_confidence,negativereason,negativereason_confidence,airline,airline_sentiment_gold,name,
#negativereason_gold,retweet_count,text,tweet_coord,tweet_created,tweet_location,user_timezone
input_file= 'c:/users/chaitanya belwal/.datasets/nlp/AirlineTweets.csv'


#The csv file has to be read in VSCode and then saved as utf-8 encoding
#else this will result in error
origDf = pd.read_csv(input_file,encoding='ISO-8859-1')

#Exclude all columns except 2, best way is to make a copy
df = origDf[['airline_sentiment', 'text']].copy()

#remove all neutral labels
df = df[df['airline_sentiment'] != 'neutral'].copy()

#Now map
df['target'] = df['airline_sentiment'].map({'positive': 0, 'negative':1})

# split up the data
df_train, df_test = train_test_split(df)

Ytrain = df_train['target']
Ytest = df_test['target']


vectorizer = TfidfVectorizer(decode_error='ignore',norm=None,max_features=2000) #norm=None will not normalize
#scikit documentation
#fit(raw_documents[, y]): Learn vocabulary and idf from training set. Calculates counts etc.
#fit_transform(raw_documents[, y]):Learn vocabulary and idf, return document-term matrix.
#transform(raw_documents): Transform documents to document-term matrix.
Xtrain = vectorizer.fit_transform(df_train['text'])
Xtest = vectorizer.transform(df_test['text']) #Do not fit the test data, only applies transform

#Xtest and Xtrain are sparse matrix
# data must not be sparse matrix before passing into tensorflow
Xtrain = Xtrain.toarray()
Xtest = Xtest.toarray()

#------------ Now build the model
D = Xtrain.shape[1]
i = Input(shape=(D,))
x = Dense(1)(i)

model = Model(i,x)

model.compile(
  loss=BinaryCrossentropy(from_logits=True), #Use BinaryCrossEntropyLoss
  optimizer=Adam(learning_rate=0.01),
  metrics=['accuracy']
)

r = model.fit(
  Xtrain, Ytrain,
  validation_data=(Xtest, Ytest), #Will print results in validation as test proceeds.
  epochs=40,
  batch_size=128,
)

# Plot loss per iteration
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
#plt.show()

# Plot accuracy per iteration
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
#plt.show()

#------------------ Do Predictions
#logit is logistic function and also caleld log-odds
#logit is inverse of sigmoid function
#the logit is a type of function that maps probability values from 
#(0,1) to (-∞,+∞)
Ptrain = model.predict(Xtrain) #This will output logits
#convert logis to prob,
Ptrain = (Ptrain > 0) * 1.0 #Convert logical to number
print(Ptrain)
Ptrain = Ptrain.flatten() #Convert to 1D array
print(Ptrain)
Ptest = (model.predict(Xtest)>0 * 1.0).flatten()

#Plot confusion matrix for training
print("Confusion matrix for training")
cm = confusion_matrix(Ytrain, Ptrain, normalize='true')
print(cm)
#Plot confusion matrix for test
print("Confusion matrix for test")
cm = confusion_matrix(Ytest, Ptest, normalize='true')
print(cm)

#print layers
print(model.layers)

word_index_map = vectorizer.vocabulary_

w = model.layers[1].get_weights()[0]
#print weights
print(w)

threshold = 1.5 #hard coding this is not the best way

print("Most positive words, as they add max weights to +ve sentiment:")
word_weight_tuples = []
for word, index in word_index_map.items():
    weight = w[index, 0]
    if weight > threshold:
        word_weight_tuples.append((word, weight))

word_weight_tuples = sorted(word_weight_tuples, key=lambda x: -x[1])
#print top ten
for i in range(10):
  word, weight = word_weight_tuples[i]
  print(word, weight)