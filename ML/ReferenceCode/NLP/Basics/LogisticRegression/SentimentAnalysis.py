
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt

import pandas as pd
from wordcloud import WordCloud

#Dataset from: https://www.kaggle.com/crowdflower/twitter-airline-sentiment

#--------------- Basic execution settings
useTestDataFile = False
useTFIDFVectorizer = True
#----------------------------------

if useTestDataFile:
    input_file= 'c:/users/chaitanya belwal/.datasets/nlp/test_spam.csv'
else:
    #tweet_id,airline_sentiment,airline_sentiment_confidence,negativereason,negativereason_confidence,airline,airline_sentiment_gold,name,
    #negativereason_gold,retweet_count,text,tweet_coord,tweet_created,tweet_location,user_timezone
    input_file= 'c:/users/chaitanya belwal/.datasets/nlp/AirlineTweets.csv'


#The csv file has to be read in VSCode and then saved as utf-8 encoding
#else this will result in error
origDf = pd.read_csv(input_file,encoding='ISO-8859-1')

#Exclude all columns except 2, best way is to make a copy
df = origDf[['airline_sentiment', 'text']].copy()

#print(df.columns) #curent column names: v1, v2
# rename columns to something better
df.columns = ['labels', 'data']

#df['labels'].hist() #plot histogram
#plt.show() #Else histogram wont show

# create binary labels
df['target'] = df['labels'].map({'positive': 0, 'neutral': 1, 'negative':2})
#print(df.head)
#Y = df['b_labels'].to_numpy()

if not useTestDataFile:
# split up the data
    df_train, df_test = train_test_split(df)
    #     df['data'], Y, test_size=0.33)# split up the data
else:
    #If using test data file ---- 
    df_train = df['data']
    df_test =df['data']

Ytrain = df_train['target']
Ytest = df_test['target']

#-- Can use tfidf vectorizer or count vectorizer
if useTFIDFVectorizer:
    featurizer = TfidfVectorizer(decode_error='ignore',norm=None,max_features=2000) #norm=None will not normalize
    Xtrain = featurizer.fit_transform(df_train['data'])
    Xtest = featurizer.transform(df_test['data'])

if not useTFIDFVectorizer:
    featurizer = CountVectorizer(decode_error='ignore')
    Xtrain = featurizer.fit_transform(df_train['data'])
    Xtest = featurizer.transform(df_test['data'])


#print(Xtrain.toarray())
#-------------- Model training
# create the model, train it, print scores
model = LogisticRegression(max_iter=500)
model.fit(Xtrain, Ytrain) #Note XTrain does not contain YTrain attributes
print("train acc:", model.score(Xtrain, Ytrain))
print("test acc:", model.score(Xtest, Ytest))

#--- F1 Score

#print("F1 score: train: ", f1_score(Ytrain, Ptrain))
#print("F1 score: test: ", f1_score(Ytest, Ptest))

#AUC Score
Prob_train = model.predict_proba(Xtrain)#[:,1]
Prob_test = model.predict_proba(Xtest)#[:,1]
print("AUC: train: ", roc_auc_score(Ytrain, Prob_train,multi_class='ovo'))
print("AUC: test: ", roc_auc_score(Ytest, Prob_test,multi_class='ovo')) #ovo = one versus one

#Confusion Matrix
Ptrain = model.predict(Xtrain)
Ptest = model.predict(Xtest)
cm = confusion_matrix(Ytrain, Ptrain)
print("Confusion Matrix on Train",cm)

cm = confusion_matrix(Ytest, Ptest)
print("Confusion Matrix on Test",cm)

#---------------- Model details
print("Model coeffs",model.coef_)


#---------------- Word index Map
wordIndexMap = featurizer.vocabulary_ #This stored the word to index mapping
print(wordIndexMap)


#---------------- Determine positive and negative words
threshold = .1
print("Most positive words:")
for word, index in wordIndexMap.items():
    weight = model.coef_[0][index]
    if weight > threshold:
        print(word, weight)