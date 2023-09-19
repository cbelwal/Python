#Parts Of Speech Tagging (POS Tagging): Sets if a word is noun, adjective etc

import nltk
from nltk.corpus import wordnet
#nltk.download("averaged_perceptron_tagger")

sentence = "Sample sentence tp help better understand".split()
words_and_tags = nltk.pos_tag(sentence)
print("POS tagging of the above words:")
print(words_and_tags) #Will assign where they are noun, adjectives etc
#NN = Noun, 

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN