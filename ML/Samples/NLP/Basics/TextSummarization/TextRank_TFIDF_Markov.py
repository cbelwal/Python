import nltk
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import textwrap
from sklearn.metrics.pairwise import cosine_similarity

from numpy import dot
from numpy.linalg import norm

#If stopwords are not downloaded
nltk.download('stopwords')

#Dataset from: https://www.kaggle.com/shivamkushwaha/bbc-full-text-document-classification
input_file= 'c:/users/chaitanya belwal/.datasets/nlp/bbc_text_cls.csv'

df = pd.read_csv(input_file)
labels = set(df['labels'])
label = 'business'


texts = df[df['labels'] == label]['text']
article = textwrap.fill(texts[0],fix_sentence_endings=True)

documents = nltk.sent_tokenize(article) #each sentence is a document
#TODO: Need to remove title from article
documents = documents[1:] #remote 1st entry as it is title
print("document: Size",len(documents)) #documents is a list

# create a tf-idf vectorizer object
#norm = L1 Norm insured there is no bias from longer sentences as they have more words
tfidf = TfidfVectorizer(stop_words=stopwords.words('english'), norm='l1')
# create a data matrix from the overviews
X = tfidf.fit_transform(documents)
S = cosine_similarity(X)

#*** Cosine Similarity Example
#A = [1,3,5]
#B = [3,6,7]
#
# cos(A.B) =A.B/|A|.|B|
#|A| = sqrt(1^2+3^2+5^2) = sqrt(2+9+25) = sqrt(36) = 6
#|B| = sqrt(3^2+6^2+7^2) = sqrt(9+36+49) = sqrt(94) = 9.69
# A.B = (1.3 + 3.6 + 5.7) = (3 + 18 + 35) = 56
# A.B/|A|.|B| = 56 / (6 * 9.69) = 56/58.14 = .96
#Note cosine_similarity(A,B) will not work as it expected a matrix
#***

#Convert S into probability so each row sums = 1

S /= S.sum(axis=1,keepdims=True)

# S matrix is similar to the Google matrix which has probability of visiting web pages
# Eigenvectors will be computed for the S matrix

#Uniform Matrix for smoothing
U = np.ones_like(S)/len(S) #Return an array of ones with the same shape and type as a given array.

#U is needed to remove 0 and make every value +ve as per
# requirements of the Perron-Forbenius Theorem  

#newS = a.S + (1-a).U: This is called convex combination
#a is the factor and new S complies with the Perron-Forbenius Theorem

factor = 0.15
S = (1 - factor) * S + factor * U

# find the limiting / stationary distribution
# this is given by the eigenvalues and eigenvecs
# we choose the eigenvecs that sum to 1

#The number of 'link's between one sentence to another is the cosine similarity
#between the two

eigenvals, eigenvecs = np.linalg.eig(S.T) #.T is the transpose of the matrix S
#Not sure why we have done a transpose above

# A.v = A.lambda
# If there is a eigenvals that is 1, then the matrix satisfies the
# Perron-Forbenius Theorem and has a limiting stationary distribution
print("All eigenvalues",eigenvals)
#
# A(i,j) = p(s_t+1 = j | s_t = i), prob of going from any state at time t, to another
# other state at time t+1
#
# A is a MxM matrix where M = number of states
#
# In PageRank, A is the prob. of going from one page to another. E.g. if a page has 2 outgoing links
# then probability for each link is 1/2
# p(s_t)   = state distribution that tells us the prob. of being in a state at time t
# p(s_t+1) = p(s_t).A
#          = p(s_t-1).A.A
#          = p(s_t-2).A.A.A
# p(s_inf) = p(s_inf-1).A
#          = p(s_inf-2).A.A
#          ...
#          = p(s_0).A.A.A......inf time, this is the limiting distribution
#Also,
# p(s_inf) = p(s_inf-1).A = p(s_inf).A, as, inf = inf - 1
# or,
# p(s_inf) = p(s_inf).A ...(1)
#
# eq.(1) is final form of limiting distribution, but is in the form of a stationary distribution
# stationary distribution, does not change after transition by A
#
# Since, we know lambda.v = A.v 
# => lambda.v = v.A-transpose, where v is Eigenvector
# if p(s_inf) is eigenvector and lambda = 1, then
# => v = v.A-transpose
# => p(s_inf) = p(s_inf).A-transpose ... (2) --> This is a stationary distribution
#
# For eq (2) to exist A, one of the eigenvalues of A should be 1
#
#
# The probability of landing on a page converges to a fixed distriution
# which are textrank scores. At the fixed distribution P(s_inf) = P(s_inf).A
# and we need to find P(s_inf)
# Perron-Frobenius theorem answers the following hold true, asssuming A is Markov (stoicastic)
# matrix, and we can travel from any state to another with +ve probability:
#
# - stationary and limiting distribution are the same
# - it is unique
# - Eigenvalue of 1 exists to being one
#
# Hence, since in the following eq. 
#   p(s_inf) = p(s_inf).A-transpose
#
# p(s_inf) is the limiting distribution, it is simply the eigenvector of A when eigenvalue is 1 

limiting_dist = eigenvecs[:,0]
print("Limiting Distribution:",limiting_dist)

#smooth out limiting distribution so they sum to 0

limiting_dist = limiting_dist/sum(limiting_dist)
print("Limiting Distribution after smoothing:",limiting_dist,"Sum:",sum(limiting_dist))


#Extra top scores in limiting_dist and then print them
sortedIdx = np.argsort(-limiting_dist) #return indices of sorted array
print("Sorted Idx:",sortedIdx)

noOfSentences = 3 #Number of sentences used to summarize
for i in range(noOfSentences): #Final summarization of the documents
    print(documents[sortedIdx[i]])




'''
#This is the manual way of computing the limiting distribution
#Starting with an initial distribution, keep multiplying by A 
# till the delta between the matrices reduces

limiting_dist = np.ones(len(S)) / len(S) #initial distribution
threshold = 1e-8
delta = float('inf')
iters = 0
while delta > threshold:
  iters += 1

  # Markov transition
  p = limiting_dist.dot(S)

  # compute change in limiting distribution
  delta = np.abs(p - limiting_dist).sum()

  # update limiting distribution
  limiting_dist = p

print(iters)
'''
