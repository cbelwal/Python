# pip install transfomers

# In this code we see various tokenizer functions and call a base model
# Note that tiken functionality is similar to the one available in tiktoken 
# library also
from transformers import AutoTokenizer

# In Hugging Face Transformers, a checkpoint typically refers to a saved 
# version of a model during training. It’s a snapshot of the model’s parameters 
# and possibly other related information at a particular point in the training 
# process. Source: https://medium.com/@sujathamudadla1213/what-is-checkpoint-in-hugging-face-transformers-d67e6b0db5b9

checkpoint = "bert-base-uncased"
# Will load the specific model weights at the specified checkpoint
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Directly calling constructor will also show token ids
print(tokenizer("Test Sentence"))

# This will show the sentence split into tokens
tokens = tokenizer.tokenize("Test Sentence")
print(tokens)

# Once tokens are known, get the token ids explicitly
ids = tokenizer.convert_tokens_to_ids(tokens)
print("Token Ids",ids)
print("Ids converted back to tokens",tokenizer.convert_ids_to_tokens(ids))
