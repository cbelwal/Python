# pip install transfomers

# This code is for Named Entity Recognition (NER) 
import numpy as np
import pandas as pd

import pickle # pickle is Python binary serialization/deserialization module
from nltk.tokenize.treebank import TreebankWordDetokenizer
from transformers import pipeline

# Set the ner pipeline
ner = pipeline("ner", aggregation_strategy='simple') #, device=0) Use if GPU available


# Load the pickle files
# This data set has text, and is tagged with the named entities
# This will be used to compare how good the ner pipeline assigns the NEs
with open('./data/ner_train.pkl', 'rb') as f:
  corpus_train = pickle.load(f)
with open('./data/ner_test.pkl', 'rb') as f:
  corpus_test = pickle.load(f)

inputs = []
targets = []

for sentence_tag_pairs in corpus_test:
  tokens = []
  target = [] # O:non-entity token,B-PER:Person,B-ORG:Organization,LOC:Location  etc. 
  for token, tag in sentence_tag_pairs:
    tokens.append(token) # word
    target.append(tag)   # Named Entity (NE)
  inputs.append(tokens)  # each inputs is a list of tokens (words)
  targets.append(target) # each targets is a list of NE matching the token

# Detokenier will combine the tokens into a sentence again
detokenizer = TreebankWordDetokenizer()

# Example of how detokenizer works
print(inputs[8]) # This prints the tokens array
detokenizedStr = detokenizer.detokenize(inputs[8])
print(detokenizedStr) # This shows the sentence with same tokens combined

# Now extract the NEs from the text
print("\n****** NER:\n",ner(detokenizedStr))



# States in NER are defined using an approach which uses BIO notation, 
# which differentiates the beginning (B) and the inside (I) of entities. 
# O is used for non-entity tokens.
# Example:
#Mark	  Watney	visited	Mars
#B-PER	I-PER	     O	  B-LOC
# -------- This function will be used to predict the output quality of the pipeline
# tokens: tokens array
# input_: token in detokenized form/raw sentence
# ner_result: predictions from transformer pipeline the pipeline
def compute_prediction(tokens, input_, ner_result):
  # map hugging face ner result to list of tags for later performance assessment
  # tokens is the original tokenized sentence
  # input_ is the detokenized string

  predicted_tags = []
  last_state = 'O' # keep track of state, so if O --> B, if B --> I, if I --> I
  current_index = 0

  # keep track of last group since the group may change
  # between consecutive entities
  # e.g. we want B-MISC -> B-PER -> I-PER
  # not          B-MISC -> I-PER -> I-PER => This is wrong as B-PER should be there before I-PER
  last_group = None

  for token in tokens:
    # find the token in the input_ (should be at or near the start)
    index = input_.find(token)
    assert(index >= 0) # token index in the main string 
    current_index += index # where we are currently pointing to

    # print(token, current_index) # debug
    # check if this index belongs to an entity and assign label
    tag = 'O'
    for entity in ner_result:
      group = entity['entity_group']
      if current_index >= entity['start'] and current_index < entity['end']:
        # then this token belongs to an entity
        if last_state == 'O':
          last_state = 'B'
        elif last_group != group:
          last_state = 'B'
        else: # If B has been seen before for same group, then it is I
          last_state = 'I'
        tag = f"{last_state}-{group}" # Combine the state with group as the test data includes this
        last_group = group
        break
    if tag == 'O':
      # reset the state
      last_state = 'O'
      last_group = None
    predicted_tags.append(tag)

    # remove the token from input_
    input_ = input_[index + len(token):]
    # update current_index
    current_index += len(token) # Return list of f"{last_state}-{group}" so that it can be compared with test tags 

  # sanity check
  # print("len(predicted_tags)", len(predicted_tags))
  # print("len(tokens)", len(tokens))
  assert(len(predicted_tags) == len(tokens))
  return predicted_tags

print("******* Predicting NER Accuracy:")
input_ = detokenizer.detokenize(inputs[8])
ner_result = ner(input_)
ptags = compute_prediction(inputs[8], input_, ner_result)

print("Prediction Tags:",ptags)


# Compute accuracy scores
from sklearn.metrics import accuracy_score, f1_score
print("Accuracy Score:",accuracy_score(targets[8], ptags)) # Compare results

'''
# The following code is commented as it takes several minutes in a CPU
# Uncomment when GPU is available
# It computes accuracy of predictions for all available data
detok_inputs = []
for tokens in inputs:
  text = detokenizer.detokenize(tokens)
  detok_inputs.append(text)

# This takes much longer in CPU than GPU
ner_results = ner(detok_inputs)

predictions = []
for tokens, text, ner_result in zip(inputs, detok_inputs, ner_results):
  pred = compute_prediction(tokens, text, ner_result)
  predictions.append(pred)

# https://stackoverflow.com/questions/11264684/flatten-list-of-lists
def flatten(list_of_lists):
  flattened = [val for sublist in list_of_lists for val in sublist]
  return flattened

# flatten targets and predictions
flat_predictions = flatten(predictions)
flat_targets = flatten(targets)

print(accuracy_score(flat_targets, flat_predictions))
print(f1_score(flat_targets, flat_predictions, average='macro'))
'''

