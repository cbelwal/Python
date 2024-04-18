#pip install sklearn
from sklearn.feature_extraction.text import CountVectorizer

text = ["Hello how are you"]
vectorizer = CountVectorizer() #built in stop words in english define stop words

token = vectorizer.fit_transform(text)
#Word to token mapping
print("Word to tokens:",vectorizer.vocabulary_)
#Token to word mapping
print("Tokens to words:",{v: k for k, v in vectorizer.vocabulary_.items()})

