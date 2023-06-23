import numpy as np
import pandas as pd
import string
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

#Dataset from: https://www.kaggle.com/shivamkushwaha/bbc-full-text-document-classification
input_file= 'c:/users/chaitanya belwal/.datasets/nlp/bbc_text_cls.csv'

df = pd.read_csv(input_file)
labels = set(df['labels'])
label = 'business'
lines=[]

texts = df[df['labels'] == label]['text']
print(texts.head())

# Will use dict to store the sparse matrix
# We will be using different dictionary keys in this work compared to the poem generator
# Also no word2idx will be used, will store words directly

#This is a test sentence
dMain = {} #Store Prob of: (a | 'this is')
dCenter={} #Store Prob of: (is | 'this a')
countMain = {} #Store count of: (* | 'this is')
countCenter = {} #Store count of: (* | 'this a')

for text in texts:
    lines = text.split('\n')
    for words in lines:
        tokens = word_tokenize(words)
        for i in range(len(tokens)-2):
            key = (tokens[i],tokens[i+2]) 
            if key not in dCenter:
                dCenter.setdefault(key,{})
                countCenter.setdefault(key,0)

            if tokens[i+1] not in dCenter[key]:
                dCenter[key][tokens[i+1]] = 0
                
            dCenter[key][tokens[i+1]] += 1
            countCenter[key] += 1

#Compute probabilities
for key in dCenter.keys():
    for key1 in dCenter[key].keys():
        dCenter[key][key1] = dCenter[key][key1]/countCenter[key]

#Read Article to spin
textToSpin = texts.iloc[0].split('\n')


np.random.seed(2345)
print("Article to spin:",textToSpin)
output = []

for line in textToSpin:
    tokens = word_tokenize(line)
    if(len(tokens) > 0):
        output.append(tokens[0])
    i=1
    while i < len(tokens) - 1:
        p = np.random.random()
        key =  (tokens[i-1],tokens[i+1]) 
        output.append(tokens[i])
        
        p = np.random.random()

        if (key in dCenter) and (p < .3): #Replacement should be prob based also
            p = np.random.random() #This is p for selecting which word to replace with
            cumu = 0
            for key1 in dCenter[key].keys():
                cumu += dCenter[key][key1]
                if p < cumu:
                    output.append('<' + key1 + '>')
                    i += 2 #Do not change next word, skip to work after it
                    break
        else:
            i += 1


output.append(tokens[len(tokens)-1])



detokenizer = TreebankWordDetokenizer()
print("**** Article after spinning\n",detokenizer.detokenize(output))


            