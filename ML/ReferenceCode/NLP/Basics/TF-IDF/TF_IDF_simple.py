import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

sampleText = []
sampleText.append("This is the first sentence that we want to test")
sampleText.append("This is the second sentence that we want to test")

df = pd.DataFrame(sampleText,columns=["text"])
print("Pandas DF")
print(df)

# create a tf-idf vectorizer object
tfidf = TfidfVectorizer(max_features=2000)

# create a data matrix from the overviews
X = tfidf.fit_transform(df["text"])

print("Value of tfidf")
print(X.toarray()) #Print as array



