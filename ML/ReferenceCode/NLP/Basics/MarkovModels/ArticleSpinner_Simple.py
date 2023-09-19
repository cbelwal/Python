import numpy as np
import matplotlib.pyplot as plt
import string
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

input_file= 'c:/users/chaitanya belwal/.datasets/nlp/test.txt'

lines=[]

with open(input_file, 'r') as fo:
    for words in fo:
        words = words.rstrip().lower()
        if words:
            words = words.translate(str.maketrans('','',string.punctuation)) #Remove punctuation
            lines.append(words)

# Will use dict to store the sparse matrix
# We will be using different dictionary keys in this work compared to the poem generator
# Also no word2idx will be used, will store words directly

#This is a test sentence
dMain = {} #Store Prob of: (a | 'this is')
dCenter={} #Store Prob of: (is | 'this a')
countMain = {} #Store count of: (* | 'this is')
countCenter = {} #Store count of: (* | 'this a')
for words in lines:
     tokens = word_tokenize(words)
     for i in range(len(tokens)):
        if(len(tokens) - i >= 3):
            if (tokens[i],tokens[i+1]) not in dMain:
                dMain.setdefault((tokens[i],tokens[i+1]),{})
                countMain.setdefault((tokens[i],tokens[i+1]),0)
            
            if tokens[i+2] not in dMain[(tokens[i],tokens[i+1])]:
                dMain[(tokens[i],tokens[i+1])][tokens[i+2]] = 0

            dMain[(tokens[i],tokens[i+1])][tokens[i+2]] += 1
            countMain[(tokens[i],tokens[i+1])] += 1

            key = (tokens[i],tokens[i+2]) 
            if key not in dCenter:
                dCenter.setdefault(key,{})
                countCenter.setdefault(key,0)

            if tokens[i+1] not in dCenter[key]:
                dCenter[key][tokens[i+1]] = 0
            
            dCenter[key][tokens[i+1]] += 1
            countCenter[key] += 1

#Compute probabilities
for key in dMain.keys():
    for key1 in dMain[key].keys():
        dMain[key][key1] = dMain[key][key1]/countMain[key]
 
for key in dCenter.keys():
    for key1 in dCenter[key].keys():
        dCenter[key][key1] = dCenter[key][key1]/countCenter[key]


textToSpin = "life is what mystery"
tokens = word_tokenize(textToSpin)

output = []
output.append(tokens[0])
i=1
while i < len(tokens) - 1:
    p = np.random.random()
    key = (tokens[i-1],tokens[i+1]) 
    output.append(tokens[i])
    if key in dCenter:
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
print(detokenizer.detokenize(output))


            