# pip install transfomers

# In this code we see various tokenizer functions and call a base model
# Note that tiken functionality is similar to the one available in tiktoken 
# library also
from transformers import AutoTokenizer

from transformers import AutoModelForSequenceClassification
# In Hugging Face Transformers, a checkpoint typically refers to a saved 
# version of a model during training. It’s a snapshot of the model’s parameters 
# and possibly other related information at a particular point in the training 
# process. Source: https://medium.com/@sujathamudadla1213/what-is-checkpoint-in-hugging-face-transformers-d67e6b0db5b9

# Will load the specific model weights at the specified checkpoint
checkpoint = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)

testSentence = "Test Sentence"

# Directly calling constructor will also show token ids
print(tokenizer(testSentence))

# This will show the sentence split into tokens
tokens = tokenizer.tokenize("Test Sentence")
print(tokens)

# Once tokens are known, get the token ids explicitly
ids = tokenizer.convert_tokens_to_ids(tokens)
print("Token Ids",ids)

# Convert the ids back to tokens
print("Ids converted back to tokens:",tokenizer.convert_ids_to_tokens(ids))

# Convert the ids back to tokens using decode function
print("Ids converted back to tokens using decode():",tokenizer.decode(ids))

# Generate the ids with encode()
# encode automatically adds the CLS and SEP token
# CLS is used in start and SEP is used at the end
ids_with_encode = tokenizer.encode("hello world")
print("Ids with encode():",ids_with_encode)

# We will be able to see the CLS and SEP id's here
print("Ids with decode:",tokenizer.decode(ids_with_encode))

# Tokenize multiple sentenves
dataMultiple = [
  "We will have nice weather today.",
  "Do you like sunny or rainy weather?",
]
print("Tokenizer Output for sentence:",tokenizer(dataMultiple))

# User same checkpoint as before
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

model_inputs = tokenizer("hello world")

# ** will unpack the items as a dictionary
# Example:
#   values = { 'a': 1, 'b': 2 }
#   s = add(**values) # equivalent to add(a=1, b=2)
# We will get an error, as we passed in list but it accepts PyTorch or TF tensors.
# Also the weights of final layers in the model are random
# outputs = model(**model_inputs)

# Create model inputs again but explicityly specify the input format 
# should be Pytorch tensors
model_inputs = tokenizer("hello world", return_tensors='pt')
print("Model inputs as PyTorch tensors:",model_inputs)

# pass the inputs again
outputs = model(**model_inputs)

# The outputs shows the logits assuming a binary classifier
# NOTE: The logits are based on an untrained model (random weights) 
# so have no meaning, yet
print("Model Output:",outputs)
print("Model Output logits (binary):",outputs.logits)

# Explicitly tell the model that there are 3 layers and
# not to assume this is a binary classifier
model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint, num_labels=3)
outputs = model(**model_inputs)

# Can also access logits property by output["logits"] -> Output treated like a dict
print("Model Output logits (tertiary classification):",outputs.logits)

# logits as numpy array
# the detach() is required to move the tensors from GPU to CPU

print("Output logits as numpy array:",outputs.logits.detach().cpu().numpy())

# Now look at processing multiple strings
# This will give an error as dataMultiple is being passed as a list
# model_inputs = tokenizer(dataMultiple, return_tensors='pt')

# To make the above work you need to pass the params as given below:
# padding: allows each input to have same number of tokens
# truncation: prevent excedding the maximum length
# 
# NOTE: PyTorch tensors need to be of same length cant be jagged array
#  
model_inputs = tokenizer(
    dataMultiple, padding=True, truncation=True, return_tensors='pt')

# Note the padding here (value of 0 in inputs)
print("Model inputs after padding and truncation:", model_inputs)

