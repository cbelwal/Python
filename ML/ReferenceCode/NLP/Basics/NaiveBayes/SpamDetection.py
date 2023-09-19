
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB

import matplotlib.pyplot as plt

import pandas as pd
from wordcloud import WordCloud

#Dataset from: https://www.kaggle.com/uciml/sms-spam-collection-dataset

#test_spam
#v1,v2,,,
#ham,This is my ham,,,
#spam,this is my spam,,,
#Note: if we use 'a' instead of 'my' it is ignored

#--------------- Basic execution settings
useTestDataFile = False
useTFIDFVectorizer = True
#----------------------------------

if useTestDataFile:
    input_file= 'c:/users/chaitanya belwal/.datasets/nlp/test_spam.csv'
else:
    input_file= 'c:/users/chaitanya belwal/.datasets/nlp/spam.csv'


#The csv file has to be read in VSCode and then saved as utf-8 encoding
#else this will result in error
df = pd.read_csv(input_file,encoding='utf-8') #encoding='ISO-8859-1'

#print(df.columns) #curent column names: v1, v2
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'],axis=1,inplace=True)
# rename columns to something better
df.columns = ['labels', 'data']

#df['labels'].hist() #plot histogram
#plt.show() #Else histogram wont show

# create binary labels
df['b_labels'] = df['labels'].map({'ham': 0, 'spam': 1})
#print(df)
Y = df['b_labels'].to_numpy()

if not useTestDataFile:
# split up the data
    df_train, df_test, Ytrain, Ytest = train_test_split(
         df['data'], Y, test_size=0.33)# split up the data
else:
    #If using test data file ---- 
    df_train = df['data']
    df_test =df['data']
    Ytrain = df['b_labels']
    Ytest = df['b_labels']


#-- Can use tfidf vectorizer or count vectorizer
if useTFIDFVectorizer:
    featurizer = TfidfVectorizer(decode_error='ignore',norm=None) #norm=None will not normalize
    Xtrain = featurizer.fit_transform(df_train)
    Xtest = featurizer.transform(df_test)
# If using test data: Xtrain for TfIdfVectorizer with norm=None:
#[[1.40546511 1.         1.         0.         1.        ]
# [0.         1.         1.         1.40546511 1.        ]]
# Reason:
#Count = [[1 1 1 0 1]
#        [0 1 1 1 1]]
#tf =  [[1/4 1/4 1/4 0/4 1/4] = [[.25 .25 .25 0 .25] 
#        [0 1/4 1/4 1/4 1/4]]    [0 .25 .25 .25 .25]]
# df[0] = 1,df[1] = 2,df[2] = 2,df[3] = 1, df[4] = 2
# idf = log ((1+n)/(1+df)) + 1 USed One Plus normalization
# idf[0] = 2+1/1+1,idf[1] = 2+1/2+1,idf[2] = 2+1/2+1,idf[3] = 2+1/1+1, idf[4] = 2+1/2+1  
# log(idf[0]) = log(3/2)+1, log(idf[1]) = log(3/3)+1,log(idf[2]) = log (3/3) + 1,log(idf[3]) = log(3/2)+1, log(idf[4]) = log(3/3) + 1

if not useTFIDFVectorizer:
    featurizer = CountVectorizer(decode_error='ignore')
    Xtrain = featurizer.fit_transform(df_train)
    Xtest = featurizer.transform(df_test)
# If using test data: Xtrain for CountVectorizer:
#[[1 1 1 0 1]
# [0 1 1 1 1]]

#print(Xtrain.toarray())
#-------------- Model training
# create the model, train it, print scores
model = MultinomialNB()
model.fit(Xtrain, Ytrain)
print("train acc:", model.score(Xtrain, Ytrain))
print("test acc:", model.score(Xtest, Ytest))

#--- F1 Score
Ptrain = model.predict(Xtrain)
Ptest = model.predict(Xtest)
print("F1 score: train: ", f1_score(Ytrain, Ptrain))
print("F1 score: test: ", f1_score(Ytest, Ptest))

#AUC Score
Prob_train = model.predict_proba(Xtrain)[:,1]
Prob_test = model.predict_proba(Xtest)[:,1]
print("AUC: train: ", roc_auc_score(Ytrain, Prob_train))
print("AUC: test: ", roc_auc_score(Ytest, Prob_test))

#Confusion Matrix
cm = confusion_matrix(Ytrain, Ptrain)
print("Confusion Matrix on Train",cm)

cm = confusion_matrix(Ytest, Ptest)
print("Confusion Matrix on Test",cm)

# visualize the data
def visualizeWordCloud(label):
  words = ''
  for msg in df[df['labels'] == label]['data']:
    msg = msg.lower()
    words += msg + ' '
  wordcloud = WordCloud(width=600, height=400).generate(words)
  plt.imshow(wordcloud)
  plt.axis('off')
  plt.show()

print("Showing wordcloud")
visualizeWordCloud("ham")