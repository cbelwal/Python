# pip install transfomers

# Pipeline to summarize text is given in this pipe
# Note that summarization can be abstractive or generative
# Abstractive - takes text from the original corpus and uses the words given there
# Generative - generates new text to add more context
import pandas as pd
import numpy as np
import textwrap

from transformers import pipeline

from pathlib import Path
homePath = str(Path.home())

# Read BBC News Articles 
df = pd.read_csv('./data/bbc_text_cls.csv')

# Extract random business articles
# Since n = 2, two business articles are extracted  
doc = df[df.labels == 'business']['text'].sample(n=2,random_state=42)
print("Total document(s):",doc.shape[0])
# Use Text wrap
summarizer = pipeline("summarization",model="sshleifer/distilbart-cnn-12-6")
with open(homePath + '\\.tokens\\HF.txt', 'r') as file:
    access_token = file.read().replace('\n', '')

# Use the more state-of-the-art model from Meta
# This model is 5G download
# Each model has a .safetensors file that contain the model weights
# CAUTION: Llama takes lot of time to load, run with GPU only
# summarizer = pipeline("summarization",model="meta-llama/Meta-Llama-3-8B",token=access_token)
# summarizer = pipeline("summarization",model="microsoft/Phi-3-mini-128k-instruct",token=access_token)

print("*** Formatted Document:",textwrap.fill(doc.iloc[0], 
                                replace_whitespace=False, 
                                fix_sentence_endings=True))
print("\n\n*** Summarized Document:",summarizer(doc.iloc[0].split("\n", 1)[1]))