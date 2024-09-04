import pandas as pd
import matplotlib.pyplot as plt
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

df = pd.read_csv('./data/tmdb_5000_movies.csv')
print(df.head)

print("All columns of row 0:")
print(df.iloc[0]) #Show all columns of row 0

x = df.iloc[0]
print(df.iloc[0]["genres"]) #show valye of a specific column

j = json.loads(df.iloc[0]["genres"])

extractG = ''
for jj in j:
    extractG = extractG + ' ' + jj['name'].replace(' ','')

print(extractG)

def conver_genres_keywords_to_str(row): #each df row
    j = json.loads(row["genres"])
    sGenres = ''
    for jj in j:
        sGenres = sGenres + ' ' + jj['name'].replace(' ','')
    
    j = json.loads(row["keywords"])
    sKeywords = ''
    for jj in j:
        sKeywords = sKeywords + ' ' + jj['name'].replace(' ','')
    
    return  "%s %s" % (sGenres,sKeywords) #This will format the string 

print("Creating new string column ...")
#Apply lambda to each row, with axis = 1
df["string"] = df.apply(conver_genres_keywords_to_str,axis = 1) #Add new column

print("Top rows after adding column")
print(df.head()["string"]) 

# ------------ Model building
# create a tf-idf vectorizer object
tfidf = TfidfVectorizer(max_features=2000)

# create a data matrix from the overviews
X = tfidf.fit_transform(df['string'])

# generate a mapping from movie title -> index (in df)
# This index will match the rows in TF-IDX
movie2idx = pd.Series(df.index, index=df['title'])
#print(movie2idx)

idx = movie2idx['Scream 3']
query = X[idx]

print(query.toarray().size) #Show all columns for movie Scream 3

scores = cosine_similarity(query, X)
print(scores)

# currently the array is 1 x N, make it just a 1-D array
scores = scores.flatten()
plt.plot(scores)
plt.show()

#Reverse Sort by order of idex
print((-scores).argsort()) #array([1164, 3902, 4628, ..., 1714, 1720, 4802])
plt.plot(scores[(-scores).argsort()])
plt.show()
# get top 5 matches
# exclude self (similarity between query and itself yields max score)
recommended_idx = (-scores).argsort()[1:6]
# convert indices back to titles
print(df['title'].iloc[recommended_idx])

#Write a function for any movie name