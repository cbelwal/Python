import pandas as pd
import numpy as np
import nltk

from nltk import word_tokenize

df = pd.read_csv('./Basics/data/bbc_text_cls.csv')

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
        doc_as_int.append(word2idx[token])
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
    i += 1
    for j in doc_as_int:
        tf[i,j] += 1

# compute IDF --------- Using Numpy
#print(tf > 0)
document_freq = np.sum(tf > 0, axis=0) # document frequency (shape = (V,))
idf = np.log(NDocs / document_freq)
#print(document_freq)
#exit()

# compute IDF --------- Manually
#document_freq = [0] * NWords #Will contains number of document having the word
#idf = [0] * NWords
#CAUTION: document_freq is not total number of occurenaces from each document
#for j in range(NWords):
#    for i in range(NDocs): 
#        if(tf[i,j] > 0): #if the word exists in this document
#            document_freq[j] += 1
#    idf[j] = np.log(NDocs/document_freq[j])


print("Document Freq",document_freq)
print("idf",idf)

# compute TF-IDF
tf_idf = tf * idf

# pick a random document, show the top 5 terms (in terms of tf_idf score)
i = np.random.choice(NDocs)
row = df.iloc[i]
print("Label:", row['labels'])
print("Text:", row['text'].split("\n", 1)[0])
print("Top 5 terms:")

scores = tf_idf[i]
print(scores < 1)

indices = (-scores).argsort()

for j in indices[:5]:
  print(idx2word[j])

         
