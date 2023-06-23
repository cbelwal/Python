from sklearn.feature_extraction.text import CountVectorizer
import numpy as py


text = ["The quick brown fox jumped over the lazy dog."]
# create the transform
vectorizer = CountVectorizer(lowercase=True)
#vectorizer = CountVectorizer(analyzer="word") #word based tokenizer
#vectorizer = CountVectorizer(analyzer="char") #char based tokenizer
#vectorizer = CountVectorizer(stop_words = "english") #built in stop words in english define stop words

'''
#nltk library: NLTK: Natural Language ToolKit
import nltk 
nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords.words('english')
'''

# tokenize and build vocab
#vectorizer.fit() #Does not return value
text = vectorizer.fit_transform(text)
token = vectorizer.transform(text)
print(token)

# summarize
print("Vectorizer Vocab:")
print(vectorizer.vocabulary_)
# encode document
vector = vectorizer.transform(text)
#print(vector)
# summarize encoded vector
#print(vector.shape)
#print(type(vector))
print(vector.toarray())

