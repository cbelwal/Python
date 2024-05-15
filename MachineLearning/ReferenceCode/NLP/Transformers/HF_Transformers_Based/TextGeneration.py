# pip install transfomers

# Text Generation using HF library
# Built with ref. to the Transformers course in Udemy offered by TheLazyProgrammer.
# Will use the robert_frost.txt file, available in data folder

from transformers import pipeline, set_seed

import numpy as np
import matplotlib.pyplot as plt


# read every line, strip of spaces
lines = [line.rstrip() for line in open('./data/robert_frost.txt')]
# remove every line that is empty
lines = [line for line in lines if len(line) > 0]

# possible model params are defined here: https://huggingface.co/models
# with model param wll default to openai-community/gpt2 and revision 6c0e608
# Model is stored in path: 
# ~\.cache\huggingface\hub\models--openai-community--gpt2\snapshots\607a30d783dfa663caf39e06633721c8d4cfcd7e
# The .safetensors file contains the model weights, while tokenzier.json and vocab.json contain the token vocab
gen = pipeline("text-generation",model="openai-community/gpt2") 

# Set seed for the transformers library
set_seed(1234)

print("Initial line:",lines[0])
print(gen(lines[0])) # generate more text based on the pipeline

# Specifiy max acceptable length of output to 20
# This output will be different as text generation is stoicastic
print(gen(lines[0],max_length=20)) # generate more text based on the pipeline

# This will return 3 results
print("Generating 3 lines:",gen(lines[0], num_return_sequences=3, max_length=20))

# Generic text generation on any prompt
prompt = "There was a person in the city of Delhi,"
print(gen(prompt, num_return_sequences=3, max_length=100))