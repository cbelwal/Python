import nltk
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import textwrap

#If stopwords are not downloaded
nltk.download('stopwords')

#Dataset from: https://www.kaggle.com/shivamkushwaha/bbc-full-text-document-classification
input_file= 'c:/users/chaitanya belwal/.datasets/nlp/bbc_text_cls.csv'

df = pd.read_csv(input_file)
labels = set(df['labels'])
label = 'business'


texts = df[df['labels'] == label]['text']
article = textwrap.fill(texts[0],fix_sentence_endings=True)

documents = nltk.sent_tokenize(article) #each sentence is a document
#TODO: Need to remove title from article
documents = documents[1:] #remote 1st entry as it is title
print("document: Size",len(documents)) #documents is a list

# create a tf-idf vectorizer object
#norm = L1 Norm insured there is no bias from longer sentences as they have more words
tfidf = TfidfVectorizer(stop_words=stopwords.words('english'), norm='l1')
# create a data matrix from the overviews
X = tfidf.fit_transform(documents)

print("X: Rows:",X.shape[0],"Cols:",X.shape[1]) #X is scipy.sparse

nDocs = X.shape[0]
nFeatures = X.shape[1]

dictDocs = {}

for i in range(nDocs):
    #count = 0
    #sum = 0
    #for j in range(nFeatures):
    row = X[i,:] #Contains all Features
    tmp = row[row != 0] #Only select values which are not zero
    avg =  tmp.mean()
        #if X[i,j] != 0:
        #    count += 1
        #    sum += X[i,j]
        
    #avg = sum /count
    dictDocs[i] = avg

print("Unsorted Dict",dictDocs)
sortedDictDocs = sorted(dictDocs.items(),key=lambda item:item[1],reverse=True)
print("Sorted Dics",sortedDictDocs)

#feature_names = tfidf.get_feature_names_out()
#print("All Features",feature_names)

#Print top 3 sentences to summarize
count = 0
for key,value in sortedDictDocs:
    print(documents[key])
    count += 1
    if(count >= 3):
        break


# Random test for list filtering
#tmpL = [0,1,5,6,0]
#t = tmpL[tmpL != 0]
#print(t)

#compute average of each row for non-zero values


