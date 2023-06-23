import pandas as pd
import numpy as np
import nltk

from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

#This routine implements a TF-IDF routine from scratch without library
#IDF =Log[(# Number of documents) / (Number of documents containing the word)] and
#TF = (Number of repetitions of word in a document) / (# of words in a document)

df = pd.DataFrame({'text':["This is text is one","This is text is two"]}) #Temp dataframe wth col name 'text'

idx = 0
word2idx = {}
idx2word={}
tokenized_docs=[]
for docRow in df['text']:
    tokens = word_tokenize(docRow.lower())
    doc_as_int = []
    for token in tokens:
        if token not in word2idx:
            word2idx[token] = idx
            idx2word[idx] = token
            idx += 1
        doc_as_int.append(word2idx[token]) #Store every integer for the word occuring ind oct
    tokenized_docs.append(doc_as_int)

# number of documents
NDocs = len(df['text'])

# Number of words
NWords = len(word2idx)

# instantiate term-frequency matrix
# note: could have also used count vectorizer
tf = np.zeros((NDocs, NWords)) # 2D array

i=-1
for doc_as_int in tokenized_docs:
    i += 1 #document idx
    for j in doc_as_int:
        tf[i,j] += 1 #will be appended every time work appears in document


# *** Dry Run output
#       : "0     1  2    1  3"
#0: doc1: "This is text is one"
#       : "0    1   2   1   4"
#1: doc2: "This is text is two"
# tf[0,0] = 1, tf[0,1] = 2, tf[0,2] = 1, tf[0,3] = 1, tf[1,4] = 0
# tf[1,0] = 1, tf[1,1] = 2, tf[1,2] = 1, tf[1,3] = 0,tf[1,4] = 1 
# Nwords = 5

# compute IDF --------- Manually
document_freq = [0] * NWords #Will contains number of document having the word
idf = [0] * NWords
#CAUTION: document_freq is not total number of occurenaces from each document
for j in range(NWords):
    for i in range(NDocs): 
        if(tf[i,j] > 0): #if the word exists in this document
            document_freq[j] += 1
    #The IDF Computation below is what is used by the TfidfVectorizer class
    idf[j] =  np.log((NDocs+1)/(document_freq[j]+1)) + 1 
    #TfidfVectorizer class does not use the following computation
    #np.log(NDocs/document_freq[j])

# *** Dry Run output
#       : "0     1  2    3  4"
#0: doc1: "This is text is one"
#       : "0    1   2   3   4"
#1: doc2: "This is text is two"
# document_freq[0] = 2, document_freq[1] = 2,document_freq[2] = 2,document_freq[3] = 1,document_freq[4] = 1
# idf[0] = log(2/2),idf[1] = log(2/2),idf[2] = log(2/2),idf[3] = log(2/1),idf[4] = log(2/1) 
# Nwords = 5 

print("Document Freq",document_freq)
print("idf",idf)

#Compute tf-idf
tfidf = np.zeros((NDocs, NWords)) # 2D array

for i in range(NDocs):
    for j in range(NWords):
        tfidf[i,j] = tf[i,j] * idf[j]

print("TF-IDF scores from build-from-scratch",tfidf)
#********************** Using library
# create a tf-idf vectorizer object
#norm=None is important else each document is normalized so that the euclidian length of each document vector equals 1
tfidf = TfidfVectorizer(max_features=5,norm=None) 
# create a data matrix from the overviews
X = tfidf.fit_transform(df["text"])
print("TF-IDF scores from library",X.toarray())

         
