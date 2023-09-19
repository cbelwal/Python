import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import textwrap

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

#Dataset from: https://www.kaggle.com/shivamkushwaha/bbc-full-text-document-classification
input_file= 'c:/users/chaitanya belwal/.datasets/nlp/bbc_text_cls.csv'

stopWords = set(stopwords.words('english')) #Download stop works

#Add more manual stop words

stopWords = stopWords.union({
    'said', 'would', 'could', 'told', 'also', 'one', 'two',
    'mr', 'new', 'year', 
})
stopWords = list(stopWords) #newer versions of CountVectorizer need this

df = pd.read_csv(input_file)

vectorizer = TfidfVectorizer(stop_words=stopWords)

X = vectorizer.fit_transform(df['text'])


#Use NMF with standard params
nmf = NMF(
    n_components=15, # Number of topics
    beta_loss="kullback-leibler",
    solver='mu',
    # alpha_W=0.1,
    # alpha_H=0.1,
    # l1_ratio=0.5,
    random_state=0,
)

nmf.fit(X)

print("All Topics and their features/words")
n_top_words = 10 
feature_names = vectorizer.get_feature_names_out() #All words
for topic_idx, topic in enumerate(nmf.components_): #Topics by Words
    top_features_ind = topic.argsort()[: -n_top_words - 1 : -1]
    top_features = [feature_names[i] for i in top_features_ind]
    print("*** Topic:",str(topic_idx))
    print("Top Features:",top_features)

Z = nmf.transform(X) #Documents by topics

#Note article can be in different topics, and prob. of them sum to 1
np.random.seed(0)
i = np.random.choice(len(df))
z = Z[i]
topics = np.arange(15) + 1
print("All Z",Z)