import nltk
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD

''' Uncomment if following not downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
'''

#input_file= '../data/tmdb_5000_movies.csv'
input_file= 'c:/users/chaitanya belwal/.datasets/nlp/tmdb_5000_movies.csv'

df = pd.read_csv(input_file)
print(df.head)
titles = df.iloc[:]["original_title"]
print(titles)

stopWords = set(stopwords.words('english')) #Download stop words
wordnet_lemmatizer = WordNetLemmatizer()

# some more stop words
stops = stopWords.union({'news', 'great'})

#Custom tokenizer 
def customTokenizer(s):
  # downcase
  s = s.lower()

  # split string into words (tokens)
  tokens = nltk.tokenize.word_tokenize(s)
  # remove short words, as they are not useful
  tokens = [t for t in tokens if len(t) > 2]
  # put words into base form
  tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens]
  # remove stopwords
  tokens = [t for t in tokens if t not in stops]
  # remove any digits, i.e. "3rd edition"
  tokens = [t for t in tokens if not any(c.isdigit() for c in t)]
  return tokens

#User vectorizer with custom tokenizer
#binary=True, will not count the tokens but only if they appear(1) or not (0)
vectorizer = CountVectorizer(binary=True, tokenizer=customTokenizer) 
X = vectorizer.fit_transform(titles) #Focus only on movie titles

# transpose X to make rows = terms, cols = documents
# this makes is a terms x document matrix
X = X.T

svd = TruncatedSVD(n_components=3) #TruncatedSVD removes noise columns
# svd is N x K matrix, where K is the topics and eq. to number of components
Z = svd.fit_transform(X) #Z is the unknown

index_word_map = vectorizer.get_feature_names_out()
#Z will contain all the important words, the less important words
#are denoised and removed.
#Plotting makes this into a zoomable plot that shows the important words
#The X and Y distribution did not make full sense
import plotly.express as px

fig = px.scatter(x=Z[:,0], y=Z[:,1], text=index_word_map, size_max=60)
fig.update_traces(textposition='top center')
fig.show()