import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet
from nltk import word_tokenize

class StemTokenizer:
  def __init__(self):
    self.porter = PorterStemmer()

  def __call__(self, doc):
    tokens = word_tokenize(doc)
    result = []
    for t in tokens:
        result.append(self.porter.stem(t))
    return result
  

if __name__ == "__main__":
  stemmer = StemTokenizer()
  print(stemmer("I am running"))