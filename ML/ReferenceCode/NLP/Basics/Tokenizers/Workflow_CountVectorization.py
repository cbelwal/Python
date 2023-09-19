import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import wordnet

from LemmaTokenizer import LemmaTokenizer
from StemTokenizer import StemTokenizer

#------------
#nltk.download("wordnet")
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('omw-1.4')
#---------

df = pd.read_csv('./Basics/data/bbc_text_cls.csv')
#metadata
print(df.head()) #top 5 records
print(len(df))
#----------
inputs = df['text']
labels = df['labels']

labels.hist(figsize=(10, 5));

#------------- Split data
inputs_train, inputs_test, Ytrain, Ytest = train_test_split(
    inputs, labels, random_state=123) #25/75 split

vectorizer = CountVectorizer()

#-----------------------
print("*** Basic model training")
Xtrain = vectorizer.fit_transform(inputs_train)
Xtest = vectorizer.transform(inputs_test) #does not give error due to previous transform
print("Shapes",Xtrain.shape,Xtest.shape)
model = MultinomialNB()
model.fit(Xtrain, Ytrain)
print("train score:", model.score(Xtrain, Ytrain))
print("test score:", model.score(Xtest, Ytest))

# with stopwords
print("*** Model training with stopwords")
vectorizer = CountVectorizer(stop_words='english')
Xtrain = vectorizer.fit_transform(inputs_train)
Xtest = vectorizer.transform(inputs_test)
print("Shapes",Xtrain.shape,Xtest.shape) #Lower number of columns than basic
model = MultinomialNB()
model.fit(Xtrain, Ytrain)
print("train score:", model.score(Xtrain, Ytrain))
print("test score:", model.score(Xtest, Ytest))

# with lemmatization
print("*** Model training with Lemmatization")
vectorizer = CountVectorizer(tokenizer=LemmaTokenizer())
Xtrain = vectorizer.fit_transform(inputs_train)
Xtest = vectorizer.transform(inputs_test)
print("Shapes",Xtrain.shape,Xtest.shape) #Lower number of columns than basic
model = MultinomialNB()
model.fit(Xtrain, Ytrain)
print("train score:", model.score(Xtrain, Ytrain))
print("test score:", model.score(Xtest, Ytest))

# with lemmatization
print("*** Model training with Stemming")
vectorizer = CountVectorizer(tokenizer=StemTokenizer())
Xtrain = vectorizer.fit_transform(inputs_train)
Xtest = vectorizer.transform(inputs_test)
print("Shapes",Xtrain.shape,Xtest.shape) #Lower number of columns than basic
model = MultinomialNB()
model.fit(Xtrain, Ytrain)
print("train score:", model.score(Xtrain, Ytrain))
print("test score:", model.score(Xtest, Ytest))