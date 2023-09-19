from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import word_tokenize
import nltk

class LemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()
  
    def get_wordnet_pos(self,treebank_tag):
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN


    def __call__(self, doc): #Default function
        tokens = word_tokenize(doc)
        words_and_tags = nltk.pos_tag(tokens)
 
        answer = []
        for word, tag in words_and_tags:
            lemma = self.wnl.lemmatize(word, pos=self.get_wordnet_pos(tag))
            answer.append(lemma)
        return answer

        #return [self.wnl.lemmatize(word, pos=self.get_wordnet_pos(tag)) \
        #        for word, tag in words_and_tags]