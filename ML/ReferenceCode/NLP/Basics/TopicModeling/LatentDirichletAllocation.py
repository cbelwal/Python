import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import textwrap

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

#Dataset from: https://www.kaggle.com/shivamkushwaha/bbc-full-text-document-classification
input_file= 'c:/users/chaitanya belwal/.datasets/nlp/bbc_text_cls.csv'



#nltk.download('stopwords')

stopWords = set(stopwords.words('english')) #Download stop words

#Add more manual stop words

stopWords = stopWords.union({
    'said', 'would', 'could', 'told', 'also', 'one', 'two',
    'mr', 'new', 'year', 
})
stopWords = list(stopWords) #newer versions of CountVectorizer need this

df = pd.read_csv(input_file)
vectorizer = CountVectorizer(stop_words=stopWords)

X = vectorizer.fit_transform(df['text'])

#LDA Top modeling for words
lda = LatentDirichletAllocation(
    n_components=15, # Number of topics
    random_state=12345,)

lda.fit(X) #lda.components_ contains the words

print("All Topics and their features/words")
n_top_words = 10 
feature_names = vectorizer.get_feature_names_out() #All words
for topic_idx, topic in enumerate(lda.components_): #Topics by Words
    top_features_ind = topic.argsort()[: -n_top_words - 1 : -1]
    top_features = [feature_names[i] for i in top_features_ind]
    print("*** Topic:",str(topic_idx))
    print("Top Features:",top_features)

Z = lda.transform(X) #Documents by topics

#Note article can be in different topics, and prob. of them sum to 1
np.random.seed(0)
i = np.random.choice(len(df))
z = Z[i]
topics = np.arange(15) + 1
print("All Z",Z)

''' Plotting routines
fig, ax = plt.subplots()
ax.barh(topics, z)
ax.set_yticks(topics)
ax.set_title('True label: %s' % df.iloc[i]['labels']);

'''


                             
