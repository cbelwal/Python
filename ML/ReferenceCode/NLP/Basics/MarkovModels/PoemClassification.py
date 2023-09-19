#Data Sources
#https://raw.githubusercontent.com/lazyprogrammer/machine_learning_examples/master/hmm_class/edgar_allan_poe.txt
#https://raw.githubusercontent.com/lazyprogrammer/machine_learning_examples/master/hmm_class/robert_frost.txt

import numpy as np
import matplotlib.pyplot as plt
import string
from sklearn.model_selection import train_test_split
from nltk import word_tokenize

input_files=['c:/users/chaitanya belwal/.datasets/nlp/edgar_allan_poe.txt',
             'c:/users/chaitanya belwal/.datasets/nlp/robert_frost.txt']

tokens = []
lines=[]
labels=[]
for label,file in enumerate(input_files):
    print(f"{file} corresponds to label {label}")
    with open(file, 'r') as fo:
        for line in fo:
            line = line.rstrip().lower()
            if line:
                line = line.translate(str.maketrans('','',string.punctuation))
                lines.append(line)
                labels.append(label)

#train test split
train_text, test_text, Ytrain, Ytest = train_test_split(lines, labels)

#tokenize the training set and convert word to index
word2idx = {}
idx=0

for line in train_text:
     tokens = word_tokenize(line)
     for token in tokens:
        if token not in word2idx:
            word2idx[token] = idx
            idx += 1

for line in test_text:
     tokens = word_tokenize(line)
     for token in tokens:
        if token not in word2idx:
            word2idx[token] = idx
            idx += 1
        
#print(word2idx)

#Convert lines into numberic format
train_lines_int = []
test_lines_int = []

for line in train_text:
    tokens = word_tokenize(line)
    line_as_int = [word2idx[token] for token in tokens]
    train_lines_int.append(line_as_int)
    #Note labels idx will remain the same

for line in test_text:
    tokens = word_tokenize(line)
    line_as_int = [word2idx[token] for token in tokens]
    test_lines_int.append(line_as_int)

print(train_lines_int[0:5])

#------ Markov transition Matrix
# initialize A and pi matrices - for both classes
V = len(word2idx)

#np.ones will setup a matrix with size V,V filled with 1s
#1 is added for Add-One Smoothing
#A[0] will be for label 0
#A[1] will be for label 1
A = []
pi = []

A.append(np.ones((V, V))) #label 0
pi.append(np.ones(V))

A.append(np.ones((V, V))) #label 1
pi.append(np.ones(V))

# build matrix for train
for t,y in zip(train_lines_int,Ytrain):
    last_idx = 0
    for idx in t: #for each idx
        if(last_idx==0):
            pi[y][idx] += 1
            last_idx = idx
        else:
            A[y][last_idx,idx] += 1

#Now we have the counts, compute probability matrix
#axis = 0 computes the sum over the rows, giving you a total for each column; axis = 1 computes the sum across the columns, giving you a total for each row 
A[0] = A[0]/A[0].sum(axis=1, keepdims=True)
pi[0] = pi[0]/pi[0].sum()

A[1] = A[1]/A[1].sum(axis=1, keepdims=True)
pi[1] = pi[1]/pi[1].sum()

logA = []
logPi = []

#In each row, x_i,j -> x_j,j+1 shows the probability of the transition 
#               based on observations 
logA.append(np.log(A[0]))
logA.append(np.log(A[1]))

logPi.append(np.log(pi[0]))
logPi.append(np.log(pi[1]))

#Compute prior probabilities
counts = [0] * 2
counts[0] = sum(l == 0 for l in Ytrain)
counts[1] = sum(l == 1 for l in Ytrain)

probs = [0] * 2
probs[0] = counts[0]/len(Ytrain)
probs[1] = counts[1]/len(Ytrain)

logProbs = [0] * 2
logProbs[0] = np.log(probs[0])
logProbs[1] = np.log(probs[1])


#Build Prediction Class
class Classifier:
    def __init__(self, logAs, logpis, logpriors):
        self.logAs = logAs
        self.logpis = logpis
        self.logpriors = logpriors
        self.K = len(logpriors) # number of classes

    #Compute the probability of a squence of words being present in a class
    #
    def _compute_log_likelihood(self, input_, class_):
        logA = self.logAs[class_]
        logpi = self.logpis[class_]

        last_idx = None
        logprob = 0
        for idx in input_:
            if last_idx is None:
                # it's the first token
                logprob += logpi[idx]
            else:
                logprob += logA[last_idx, idx]
        
        # update last_idx
        last_idx = idx
    
        return logprob
  
    #inputs are arrays of arrays of int (for each line)
    def predict(self, inputs):
        predictions = np.zeros(len(inputs))
        for i, input_ in enumerate(inputs):
            #This is normally P(poem | author=k). p(author=k)/p(poem)
            #However since we compute argmax p(poem) can be eliminated
            #Therefore it is P(poem | author=k). p(author=k)
            #Since we consider log values
            #log(P(poem | author=k). p(author=k)) = log(P(poem | author=k)) + log(p(author=k))
            posteriors = [self._compute_log_likelihood(input_, c) + self.logpriors[c] \
                    for c in range(self.K)]
            pred = np.argmax(posteriors)
            predictions[i] = pred
        return predictions


#Make predictions
cls = Classifier(logA, logPi, logProbs)

'''
#Temp for test
label0_train_lines_int = []
for line,y in zip(train_lines_int,Ytrain):
    if y == 0:
        label0_train_lines_int.append(line)

Ptrain = cls.predict(label0_train_lines_int)
print("Training Prob:", Ptrain)
'''

#exit()
Ptrain = cls.predict(train_lines_int)
print("Training Prob:", Ptrain)
Ptest = cls.predict(test_lines_int)
print("Testing Prob:", Ptest) 

#Accuracy Metrics
from sklearn.metrics import confusion_matrix, f1_score

print("Confusion Matrix for training:",confusion_matrix(Ytrain,Ptrain))
print("Confusion Matrix for test:",confusion_matrix(Ytest,Ptest ))

print("F1 Score for train:",f1_score(Ytrain, Ptrain))
print("F1 Score for test:",f1_score(Ytest, Ptest))
