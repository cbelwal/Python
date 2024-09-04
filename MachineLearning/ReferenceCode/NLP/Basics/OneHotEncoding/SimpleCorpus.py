# Sourced from: 
# https://www.geeksforgeeks.org/one-hot-encoding-in-nlp/
import numpy as np



# Define the corpus of text
corpus = [
    "The quick brown fox jumped over the lazy dog.",
    "She sells seashells by the seashore.",
    "Peter Piper picked a peck of pickled peppers."
]
 
# Create a set of unique words in the corpus
unique_words = set()
for sentence in corpus:
    for word in sentence.split():
        unique_words.add(word.lower())
 
# Create a dictionary to map each
# unique word to an index
word_to_index = {}
for i, word in enumerate(unique_words):
    word_to_index[word] = i
 
# Create one-hot encoded vectors for
# each word in the corpus
one_hot_vectors = []
for sentence in corpus:
    sentence_vectors = []
    for word in sentence.split():
        vector = np.zeros(len(unique_words))
        vector[word_to_index[word.lower()]] = 1
        sentence_vectors.append(vector)
    one_hot_vectors.append(sentence_vectors)
 
# Print the one-hot encoded vectors 
# for the first sentence
print("One-hot encoded vectors for the first sentence:")
for vector in one_hot_vectors[0]:
    print(vector)