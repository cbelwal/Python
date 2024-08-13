# This is similar to: FineTuning_SentimentAnalysis_CustomDS.py
# except here we are using the config object to set the proper lable names
# and not use LABEL_0 and LABEL_1
#
# Will do sentiment analysis on a custom dataset
#
# This is similar to the SA task on GLUE data set, except this one
# does not use an in built dataset.
#
# Rest of the code is similar to one used in GLUE dataset

from datasets import load_dataset

import pandas as pd
import numpy as np
from os.path import expanduser

#---------- Metrics Libs
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

homeDir = expanduser("~")


#------------------- Load Data
df_ = pd.read_csv('./data/AirlineTweets.csv') # Use . as code run in root folder

# Copy the 2 relevant columns
df = df_[['airline_sentiment', 'text']].copy()

# Change the text postive, negative and neutral to integers
target_map = {'positive': 1, 'negative': 0, 'neutral': 2}
df['target'] = df['airline_sentiment'].map(target_map)

# Change the lable of 'text' to 'sentence' and 'target' to 'label'
# This is required as we will use HF library to reload the exported CSV of this dataset
# and in a sentiment analysis task the target label column should be 'label' 
df2 = df[['text', 'target']]
df2.columns = ['sentence', 'label']
# Export to CSV so we can reload it via the HF library.
df2.to_csv('./data/data.csv', index=None)

# now load the dataset using HF library
raw_dataset = load_dataset('csv', data_files='./data/data.csv')

#------------------- Prepare data into train and test
# Split 30% into test and 70% into train
dataSplit = raw_dataset['train'].train_test_split(test_size=0.3, seed=42)

print("Data split details:",dataSplit)

#-------------------- Tokenization
from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased" # Define the pretrained model checkpoint
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Tokensize 3 sentences
tokenized_sentences = tokenizer(dataSplit['train'][0:3]['sentence'])

# Print the tokenized output
# This will include only the attention_asks and the token_ids
print("Sample Tokens:", tokenized_sentences) 

# This function is used by the map function
# Truncation sets a max limit on size
# Details: Padding adds a special padding token to ensure shorter sequences will have the same length as either the longest sequence in a batch or the maximum length accepted by the model. 
# Truncation works in the other direction by truncating long sequences.
# https://huggingface.co/docs/transformers/en/pad_truncation
def tokenize_fn(batch):
  return tokenizer(batch['sentence'], truncation=True)

#-------------------- Training
# map() will call tokenize_fn() for multiple rows of data
tokenized_datasets = dataSplit.map(tokenize_fn, batched=True)

from transformers import Trainer,TrainingArguments, AutoConfig

# This command will load the config. params
config = AutoConfig.from_pretrained(checkpoint)
print("Current config setup id2label:",config.id2label)

# target_map = {'positive': 1, 'negative': 0, 'neutral': 2}
config.id2label = {v:k for k, v in target_map.items()}
config.label2id = target_map
print("After modification setup id2label:",config.id2label)

# Specify values for the training arguements class
training_args = TrainingArguments(
  output_dir='training_dir',
  evaluation_strategy='epoch',
  save_strategy='epoch',
  num_train_epochs=3,
  per_device_train_batch_size=16,
  per_device_eval_batch_size=64,
)

# AutoModelForSequenceClassification is a generic model class that will 
# be instantiated as one of the sequence classification
# model classes of the library.
#  

from transformers import AutoModelForSequenceClassification, pipeline

# from_pretrained() will load the specific model whose checkpoint is specified
# checkpoint is defined before and is:"distilbert-base-uncased"
# distilbert is a shorter version of BERT
# BERT Base has 110 million params
model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint, config=config) # Specify the updated config here, no need to specify labels as config contains it
    
# summary class is used to give more information on the model
from torchinfo import summary

print("Model type:", type(model))
print("Model summary:", summary(model))

# Model summary shows 66,955,010 params (66+MM)
# logits_and_lables is a tuple
def compute_metrics(logits_and_labels):
  logits, labels = logits_and_labels
  predictions = np.argmax(logits, axis=-1)
  acc = np.mean(predictions == labels)
  f1 = f1_score(labels, predictions, average='macro')
  return {'accuracy': acc, 'f1': f1} # Compute both the metrics


# Set trainer object
# Tokenized dataset has been defined before
# This is equivalent to model fine tuning on the validation
# data set 
# Note the compute_metrics will print metrics at each epoch
trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    )

modelPath = homeDir + '\\Models\\fine_tuned_distelbert_custom'
# Do the training -> will result in updated weights
#print("Starting Model training ...")
# Training is taking ~ 1hr10 mins in CPU
#
# During training the model is saved at each epoch in the folder
# 'training_dir' in the root folder. The folder is named after the number of 
# training steps
#
# trainer.train()

# Save the model in folder 'fine_tuned_distelbert_model'
# NOTE: Once the model is trained persist it so training is not
# needed again. 
# Default HF models are saved in: ~\.cache\huggingface\hub
# trainer.save_model(modelPath)

#------------ Comments following after model trained
# Now do inference from model after the training
newModelPipeline = pipeline('text-classification',
                      model='training_dir/checkpoint-1282')#, # Load from specific checkpoint
                      #device=1)

# newModelPipeline = pipeline('text-classification', model=modelPath) #'fine_tuned_distelbert')
# Try some sample genericsentiments
print("Test with same samples:")
# NOTE: We have not updated the config file here with more descriptive label names
# So it will still show LABEL_0 and LABEL_1
print("+ve Sentiment:",newModelPipeline('This was a very good book!'))
print("-ve Sentiment:",newModelPipeline('This was a really bad book!'))

print("Predicting in entire test set")
test_pred = newModelPipeline(dataSplit['test']['sentence']) # Only pass the sentences
print("Sample of predictions:", test_pred[0:5])

# Let us compute the scores
# Extract the nummeric values, so LABEL_0 will be 0 and LABEL_1 will be 1
def get_label(d):
  return int(d['label'].split('_')[1])

test_pred = [get_label(d) for d in test_pred]

# Call the accuracy function
print("acc:", accuracy_score(dataSplit['test']['label'], test_pred))
# Get F1
print("f1:", f1_score(dataSplit['test']['label'], test_pred, average='macro'))

import seaborn as sn
import matplotlib.pyplot as plt # Needed for the plt.show() command
# plot CM
def plot_cm(cm):
  classes = ['negative', 'positive', 'neutral']
  df_cm = pd.DataFrame(cm, index=classes, columns=classes)
  ax = sn.heatmap(df_cm, annot=True, fmt='g')
  ax.set_xlabel("Predicted")
  ax.set_ylabel("Target")
  plt.show()
  
cm = confusion_matrix(dataSplit['test']['label'], test_pred, normalize='true')
plot_cm(cm)
