#Data Sources
#https://raw.githubusercontent.com/lazyprogrammer/machine_learning_examples/master/hmm_class/robert_frost.txt

import numpy as np
import matplotlib.pyplot as plt
import string
from nltk import word_tokenize

input_file= 'c:/users/chaitanya belwal/.datasets/nlp/robert_frost.txt'
#input_file= 'c:/users/chaitanya belwal/.datasets/nlp/test.txt'

lines=[]

with open(input_file, 'r') as fo:
    for words in fo:
        words = words.rstrip().lower()
        if words:
            words = words.translate(str.maketrans('','',string.punctuation)) #Remove punctuation
            lines.append(words)
               
word2idx = {}
idx2word = {}
idx=0

for words in lines:
     tokens = word_tokenize(words)
     for token in tokens:
        if token not in word2idx:
            word2idx[token] = idx
            idx2word[idx] = token
            idx += 1
        
#print(word2idx)

#Convert lines into numeric format
lines_int = []

for words in lines:
    tokens = word_tokenize(words)
    line_as_int = [word2idx[token] for token in tokens]
    lines_int.append(line_as_int)
    #Note labels idx will remain the same

print("Sample lines",lines_int[0:5])

#------ Markov transition Sparse Matrix -> Stored in Dict

pi = {} #Start
sumPi = 0
A1 = {} #First order Markov
sumA1 = {}
A2 = {} #Second order Markov
sumA2 = {}


# build dict (substitute for Sparse Matrix) 
idxEnd = len(word2idx) + 100

for iline in lines_int:
    iline.append(idxEnd)    
    last_idx = -1
    last_idx_1 = [-1] * 2
    for idx in iline: #for each idx
        if(last_idx==-1): #first word
            if(idx not in pi):
                pi[idx] = 0
            pi[idx] += 1
            sumPi += 1
            last_idx = idx
            last_idx_1[0] = idx
        else:
            #--------------- A1
            if last_idx not in A1:
                A1[last_idx] = {}
                sumA1[last_idx] = 0 #Will not exist in sumA1 
            if idx not in A1[last_idx]:
                A1[last_idx][idx] = 0
            A1[last_idx][idx] += 1
                
            sumA1[last_idx] += 1
            last_idx = idx
            #---------------- A2
            if(last_idx_1[1] == -1):
                last_idx_1[1] = idx
            else:
                if last_idx_1[0] not in A2:
                    A2[last_idx_1[0]] = {}
                    sumA2[last_idx_1[0]] = {}
                if last_idx_1[1] not in A2[last_idx_1[0]]:
                    A2[last_idx_1[0]][last_idx_1[1]] = {}
                    sumA2[last_idx_1[0]][last_idx_1[1]] = 0
                if idx not in A2[last_idx_1[0]][last_idx_1[1]]:
                    A2[last_idx_1[0]][last_idx_1[1]][idx] = 0 
                A2[last_idx_1[0]][last_idx_1[1]][idx] += 1
                sumA2[last_idx_1[0]][last_idx_1[1]] += 1 
                last_idx_1[0] = last_idx_1[1]
                last_idx_1[1] = idx
    #end of line
    #if(idxEnd not in A2[last_idx_1[0]][last_idx_1[1]]):
    #    A2[last_idx_1[0]][last_idx_1[1]][idxEnd] = 0
    #A2[last_idx_1[0]][last_idx_1[1]][idxEnd] += 1  


#Now we have the counts, compute probability matrix
#Do not convert to log proababilities
#sum pi's
for k in pi.keys():
     pi[k] = pi[k]/sumPi #Compute log probabilities/standard prob.

#A1's
for k1 in A1.keys():
    for k in A1[k1].keys():
        A1[k1][k] = A1[k1][k]/sumA1[k1]
         
#A2's
for k2 in A2.keys():
    for k1 in A2[k2].keys():
        for k in A2[k2][k1].keys():
            A2[k2][k1][k] = A2[k2][k1][k]/sumA2[k2][k1]


#---Generate--------------
noOfLines = 6
allLines = ''
#--- Select starting char
p = np.random.random() #Will generate value between 0 and 1 

for i in range(noOfLines):
    words = []
    cumuProb = 0
    for k in pi.keys():
        cumuProb += pi[k] #Keep summing p so that it insures we reach a value
        if(p < cumuProb): #select
            s0 = k
            break

    words.append(s0)
    #---- select next char
    p = np.random.random() #Will generate value between 0 and 1 
    cumuProb = 0
    for k in A1[s0].keys():
        cumuProb += A1[s0][k]
        if(p < cumuProb): #select
            s1 = k
            break

    words.append(s1)

    #Second degree word
    
    dA2 = A2[s0][s1]
    exit = False
    print("Starting generation loop for line ", str(i), " ...")
    while(True):
        cumuProb = 0
        p = np.random.random() #Will generate value between 0 and 1 
        for k in dA2.keys():
            cumuProb += dA2[k]
            if (k == idxEnd):
                exit = True
                break
            if(p < cumuProb): #select
                words.append(k)
                dA2 = A2[s1][k] #This line gives an error some times
                s1 = k
                break
        if exit:
            break
            
    #Now print line
    sentence = ''
    for wordIdx in words:
        sentence = sentence + ' ' + idx2word[wordIdx]
    allLines = allLines + sentence + '\n'
print(allLines)
