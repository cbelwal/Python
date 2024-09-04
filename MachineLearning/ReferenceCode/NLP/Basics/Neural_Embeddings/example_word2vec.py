from gensim.models import KeyedVectors

word_vectors = KeyedVectors.load_word2vec_format(
  '~/.datasets/nlp/GoogleNews-vectors-negative300.bin',
  binary=True
)

def find_analogies(w1, w2, w3):
  # w1 - w2 = ? - w3
  # e.g. king - man = ? - woman
  #      ? = +king +woman -man
  r = word_vectors.most_similar(positive=[w1, w3], negative=[w2])
  print("%s - %s = %s - %s" % (w1, w2, r[0][0], w3))

def nearest_neighbors(w):
  r = word_vectors.most_similar(positive=[w])
  print("neighbors of: %s" % w)
  for word, score in r:
    print("\t%s" % word)

print(find_analogies('king', 'man', 'woman'))

print(nearest_neighbors('king'))
print(nearest_neighbors('france'))
