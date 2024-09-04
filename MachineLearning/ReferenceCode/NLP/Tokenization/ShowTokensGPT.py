# This file shows the token length and the token mapping used in GPT 3.5/4 models
# By changing the encoding used you can apply this to other GPT models also
# This uses the tiktoken libary which is by OpenAI so the encodings
# will only apply to OpenAI introduced models

# pip install tiktoken
# pip install openai

import tiktoken
import argparse # For argument parsing
import os

# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--Text", help = "Text to tokenize in quotes")
args = parser.parse_args()
# Encoding names and models applied to
# cl100k_base: GPT 3.5/4
# p50k_base davinci
# r50k_base: GPT 3/2

#print(os.path.join(tempfile.gettempdir(), "data-gym-cache"))
#print(os.environ["TIKTOKEN_CACHE_DIR"])
#print(os.environ["DATA_GYM_CACHE_DIR"])
# The above encoding to model maps are defined in MODEL_TO_ENCODING dict here:
# https://github.com/openai/tiktoken/blob/main/tiktoken/model.py

# Can also get encoding, like
# encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')
# "cl100k_base" is the tokenirzer used in GPT
encoding = tiktoken.get_encoding("cl100k_base")

if(args.Text):
    textToEncode = args.Text
else:
    textToEncode = "This is the default text to encode"

tokens = encoding.encode(textToEncode)

print("Number of Tokens:",len(tokens))
print("Token details:")
for token in tokens:
    print(str(token),":",encoding.decode_single_token_bytes(token)) 

