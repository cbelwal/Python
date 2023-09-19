import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import POS_Tagging

#-------------- Stemmer
porter = PorterStemmer()
print("Stemming:",porter.stem("walking")) #returns 'walk'
print("Stemming:",porter.stem("walks")) #returns 'walk'
print("Stemming:",porter.stem("bosses")) 
print("Stemming:",porter.stem("replacement")) 

#----------- Lemmatization
#nltk.download("wordnet")
#POS tags: Parts Of Speech: example, Noun, Verb, Adjective

lemmatizer= WordNetLemmatizer()
print(lemmatizer.lemmatize("mice")) #output mouse)

print(lemmatizer.lemmatize("going")) #returns 'going
print(lemmatizer.lemmatize("going",pos=wordnet.VERB)) #return 'go'

#Find parts of speech tagging
nltk.download("averaged_perceptron_tagger")

sentence = "Sample sentence to help better understand".split()
words_and_tags = nltk.pos_tag(sentence)
print(words_and_tags) #Will assign where they are noun, adjectives etc

print("Lemma of words")
for word,tag in words_and_tags:
    lemma = lemmatizer.lemmatize(word,pos=POS_Tagging.get_wordnet_pos(tag))
    print(lemma, end=" ")