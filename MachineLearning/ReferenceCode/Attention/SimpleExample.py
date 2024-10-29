# This example show attention using hard coded data
# This is not the general attention showing the orig. seq2seq paper (Bahdanau et al),
# but is the attention mech. used in transformers 

# Original Source code is given here: https://machinelearningmastery.com/the-attention-mechanism-from-scratch/

from numpy import array
from numpy import random
from numpy import dot
from scipy.special import softmax

# encoder representations of four different words
# each work has an embedding of size 3
# these embeddings are hard-coded and have integer values 
# for demonstration purposes
# Embedding size = 3 
word_1 = array([1, 0, 0]) # these are assigned in each row
word_2 = array([0, 1, 0])
word_3 = array([1, 1, 0])
word_4 = array([0, 0, 1])

print(f"word_1 shape: {word_1.shape}") # 3, -> 3 rows

# stacking the word embeddings into a single array
# The words array represents a sequence of words like in a sentence
words = array([word_1, word_2, word_3, word_4]) # convert to numpy array
print(f"words shape: {words.shape}") #4x3 -> 4 rows x 3 cols

# generating the weight matrices
# The weight matrix will be size of the embedding which is 3 in our case
# These weights will be updated during the training process
random.seed(42)
W_Q = random.randint(low=0, high=3, size=(3, 3)) # Query
W_K = random.randint(low=0, high=3, size=(3, 3)) # Key
W_V = random.randint(low=0, high=3, size=(3, 3)) # Value

# generating the queries, keys and values
# @ operator was introduced in Python 3.5 for Matmul 
Q = words @ W_Q # 4x3 @ 3x3 = 4x3 matrix, note here 4 is number of words in sentence
K = words @ W_K # 4x3 @ 3x3 = 4x3 matrix
V = words @ W_V # 4x3 @ 3x3 = 4x3 matrix

# print("Q Shape:",Q.shape) # 4x3

# scoring the query vectors against all key vectors
scores = Q @ K.transpose() # 4x3 @ (4x3)T = 4x3 @ (3x4) = 4x4 Matrix
#print("scores Shape:", scores.shape) # 4x4

# computing the weights by a softmax operation
# 0.5 takes the sq rt for # cols, take softmax along cols specified by axis=1
# This is eq. to a ANN with softmax
# weights are also represented by alpha in papers 
attention_weights = softmax(scores / K.shape[1] ** 0.5, axis=1) # 4x1 matrix
print("weights Shape:", attention_weights.shape) # 4x4 - matrix. Sum will equal to 1 along cols
print("Weights,",attention_weights)

# computing the attention by a weighted sum of the value vectors
# this attention is what is termed attwntion(q,K,V) in Vaswani et al
# In Bahdanau et al, V is the hidden state at time t', and final attention is also
# termed the context  
attention = attention_weights @ V # 4x4 @ 4x3 Matrix = 4x3 Matrix 

print(attention)

# Further processing is done after this