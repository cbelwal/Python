# To install pytroch with CUDA:
#
# pip3 install torch torchvision torchaudio --upgrade --force-reinstall --index-url https://download.pytorch.org/whl/cu121
#
# We will fine tune an existing model for Sentiment Analysis
#
# Install the datasets library
# pip install transformers datasets
#
# The General Language Understanding Evaluation (GLUE) benchmark is a 
# collection of resources for training, 
# evaluating, and analyzing natural language understanding systems.
# 
# https://gluebenchmark.com/

# Datasets is a lightweight library from HF providing two main features
# 1. One-line dataloaders for many public datasets
# 2. Efficient data pre-processing
from datasets import load_dataset

import numpy as np
from os.path import expanduser
homeDir = expanduser("~")

# Load the glue data set with sst2 subtask into raw_datasets object
# can also load the amazon_polarity datasets but that will take time

raw_datasets = load_dataset("glue", "sst2") # sst2 is the subtask

#print("Full dataset details:",raw_datasets)
#print("Train dataset details:",raw_datasets['train'])

# dir() shows what attributes and methods are there in a object
# does not show values
#print("Train dataset dir:",dir(raw_datasets['train']))

print("Train Dataset type:",type(raw_datasets['train']))
print("Train Dataset data:",raw_datasets['train'].data)

# Thw following will output:
# {'sentence': 'hide new secretions from the parental units ', 'label': 0, 'idx': }
# Showing there are 3 columns: sentence,label and idx
print("Train Dataset Single row:",raw_datasets['train'][0])

print("Train Dataset two rows:",raw_datasets['train'][1000:1002])

# This will give more details on the features
print("Features of the dataset:", raw_datasets['train'].features)

#-------------------- Tokenization
from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased" # Define the pretrained model checkpoint
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Tokensize 3 sentences
tokenized_sentences = tokenizer(raw_datasets['train'][0:3]['sentence'])

# Print the tokenized output
# This will include only the attention_asks and the token_ids
print("Tokens:", tokenized_sentences) 

# This function is used by the map function
# Truncation sets a max limit on size
def tokenize_fn(batch):
  return tokenizer(batch['sentence'], truncation=True)

#-------------------- Training
# map() will call tokenize_fn() for multiple rows of data
tokenized_datasets = raw_datasets.map(tokenize_fn, batched=True)

from transformers import TrainingArguments

# Specify values for the training arguements class
training_args = TrainingArguments(
  'my_trainer',
  evaluation_strategy='epoch',
  save_strategy='epoch',
  num_train_epochs=1,
)

# AutoModelForSequenceClassification is a generic model class that will 
# be instantiated as one of the sequence classification
# model classes of the library.
#  

from transformers import AutoModelForSequenceClassification

# from_pretrained() will load the specific model whose checkpoint is specified
# checkpoint is defined before and is:"distilbert-base-uncased"
# distilbert is a shorter version of BERT
# BERT Base has 110 million params
model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint,
    num_labels=2)

#print("Type of model:", type(model))
#print("Model details:", model)

# summary class is used to give more information on the model
from torchinfo import summary

print("Model type:", type(model))
print("Model summary:", summary(model))

# Model summary shows 66,955,010 params (66+MM)

# Store all the weights(params) before the fine-tuning
# We will then compare it to weight after training
params_before = []
prevSize = 1
total = 0
for name, p in model.named_parameters():
  # print(f"name: {name},size: {len(p)}")
  # total += len(p) * prevSize
  # prevSize = len(p)
  # The param calculation is not adding up, revisit later
  params_before.append(p.detach().cpu().numpy())

# Note: The size of params only shows the different param types
# where each contains multiple params
#print("Totals:",total)
#print("Size of params:", len(params_before))

from transformers import Trainer

from datasets import load_metric

metric = load_metric("glue", "sst2", trust_remote_code=True)

# Example of calling the metric
# This gives score of 0.66 as 2 out of 3 predictions are correct
# The last one is not correct.
print("Sample Metrics:",metric
      .compute(predictions=[1, 0, 1], references=[1, 0, 0]))

# logits_and_lables is a tuple
def compute_metrics(logits_and_labels):
  # metric = load_metric("glue", "sst2")
  logits, labels = logits_and_labels
  predictions = np.argmax(logits, axis=-1)
  return metric.compute(predictions=predictions, references=labels)


# Set trainer object
# Tokenized dataset has been defined before
# This is equivalent to model fine tuning on the validation
# data set 
trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    )

# Do the training -> will result in updated weights

#------------ Comments following after model trained
#'''
print("Starting Model training ...")
# Training is taking ~ 1hr10 mins in CPU
#
trainer.train()

# Save the model in folder 'fine_tuned_distelbert_model'
# NOTE: Once the model is trained persist it so training is not
# needed again. 
# Default HF models are saved in: ~\.cache\huggingface\hub
trainer.save_model(homeDir + '\\Models\\fine_tuned_distelbert')
#------------ End Comment
#'''

# Now we can use our model via a pipepline
from transformers import pipeline, AutoModel

# Since path cannot be specified in the pipeline directly
# we have to create two variables to define it
modelPath = homeDir + '\\Models\\fine_tuned_distelbert'
modelToUse = AutoModel.from_pretrained(modelPath)
tokenizerToUse = AutoTokenizer.from_pretrained(modelPath)

# CAUTION: if model and tokenizer are specified separately the test with sample sentiments
# is failing with some error in 'logits'. 
# newmodel = pipeline('text-classification', model=modelToUse, tokenizer=tokenizerToUse,device=0)

# Use modelPath variable, specifying path in constructor is not working.
 
newModelPipeline = pipeline('text-classification', model=modelPath) #'fine_tuned_distelbert')
# Try some sample sentiments
print("Test with same samples:")
print("+ve Sentiment:",newModelPipeline('This was a very good book!'))
print("-ve Sentiment:",newModelPipeline('This was a really bad book!'))

# The output is coming label_1 and label_2 which is not what we need
# Need to change the config.json file inside model to update with new labels.
import json

# 
config_path = modelPath + '\\config.json'
with open(config_path) as f:
  j = json.load(f)

# Add a new field
j['id2label'] = {0: 'negative', 1: 'positive'}

with open(config_path, 'w') as f:
  json.dump(j, f, indent=2)

# Let's check the prediction with new labels
# Load the pipeline again

newModelPipeline = pipeline('text-classification', model=modelPath)

# Test again with new labels
print("Test with same samples after change in JSON:")
# NOTE: If JSON is already updated the labels will look the same as before.
print("+ve Sentiment:",newModelPipeline('This was a very good book!'))
print("-ve Sentiment:",newModelPipeline('This was a really bad book!'))

# Compare the delta of the params 
# Store new params
# Load the model again

params_after = []
# modelToUse is already loaded
for name, p in modelToUse.named_parameters():
  params_after.append(p.detach().cpu().numpy())

# Now take the delta between params_before and params_after

# zip join two tuples together
for p1,p2 in zip(params_before,params_after):
  diff = np.abs(p1-p2) # diff is a matrix
  print(np.sum(diff)) # Sum of differences

# The delta shows how new weights are being trained.
