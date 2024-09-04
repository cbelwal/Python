# This sample deals with classification of multiple sentences
# and also Textual Entailment
#
# From public sources:
#
# Textual entailment is a binary relation between two natural-language 
# texts (called 'text' and 'hypothesis'), where readers of the 'text' 
# would agree the 'hypothesis' is most likely true 
# (Peter is snoring → A man sleeps).
#
# Recognizing Textual Entailment is the task of determining, for example, that the sentence: "Google files for its 
# long awaited IPO" entails that "Google goes public".
#
# RTE: Recognizing Textual Entailment
# This used the training and validation dataset from GLUE
#
# The General Language Understanding Evaluation (GLUE) benchmark is a 
# collection of resources for training, 
# evaluating, and analyzing natural language understanding systems.
#
from datasets import load_dataset
import numpy as np

# The Recognizing Textual Entailment (RTE) datasets come from a series of annual
# textual entailment challenges. We combine the data from RTE1 (Dagan et al.,
# 2006), RTE2 (Bar Haim et al., 2006), RTE3 (Giampiccolo et al., 2007), and RTE5
# (Bentivogli et al., 2009).4 Examples are constructed based on news and
# Wikipedia text. We convert all datasets to a two-class split, where for
# three-class datasets we collapse neutral and contradiction into not
# entailment, for consistency.
#
# Load the rte dataset within glue
#
raw_datasets = load_dataset("glue", "rte")

# NOTE: glue datasets already provide split datasets for: train, validation and test
# print features in raw_datasets, not the columns 'sentence1' and 'sentence2'

print("Features:", raw_datasets['train'].features)
print("Some train samples:",raw_datasets['train']['sentence1'][:5])

from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased" # Define the pretrained model checkpoint
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Tokenize 2 sentences in same row
tokenized_sentences = tokenizer(raw_datasets['train']['sentence1'][0],
                                 raw_datasets['train']['sentence2'][0])

# Print the tokenized output
# This will include only the attention_masks and input_ids
# This is a dict with 2 keys ['input_ids'] and ['attention_mask']

print("Sample Tokens:", tokenized_sentences) 

# Decoded inputs (convert the ids to word reps)
print("Tokens back to word rep:", tokenizer.decode(tokenized_sentences['input_ids']))

from transformers import AutoModelForSequenceClassification, \
    pipeline, AutoConfig,Trainer, TrainingArguments

config = AutoConfig.from_pretrained(checkpoint)
print("Current config setup id2label:",config.id2label)

target_map = {'positive': 1, 'negative': 0}
# After execution of code output will be:
# Current config setup id2label: {0: 'LABEL_0', 1: 'LABEL_1'}
config.id2label = {v:k for k, v in target_map.items()}
config.label2id = target_map

# ---------- Prepare data for training

training_args = TrainingArguments(
  output_dir='training_dir',
  evaluation_strategy='epoch',
  save_strategy='epoch',
  num_train_epochs=5,
  per_device_train_batch_size=16,
  per_device_eval_batch_size=64,
  logging_steps=150, 
  # by defaults logging_steps has a very large value, and since there are very 
  # few sample in this datasets, set logging_steps to lower value else 
  # 'no log' will appear under training loss
)

def tokenize_fn(batch):
  return tokenizer(batch['sentence1'], batch['sentence2'], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_fn, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint, config=config)

#from datasets import load_metric
#metric = load_metric("glue", "rte")

from sklearn.metrics import f1_score

# Compute general metrics
def compute_metrics(logits_and_labels):
  logits, labels = logits_and_labels
  predictions = np.argmax(logits, axis=-1)
  acc = np.mean(predictions == labels)
  f1 = f1_score(labels, predictions)
  return {'accuracy': acc, 'f1': f1}

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics)

from os.path import expanduser
homeDir = expanduser("~")
modelPath = homeDir + '\\Models\\fine_tuned_distelbert_RTE'

#--------- Do not train model if its already saved
# Uncomment following lines to train model
#print("Starting training ...")
#trainer.train()
print("Training finished, saving model ...")
trainer.save_model(modelPath)

#------------------------------------

# Test the model
p = pipeline('text-classification', model=modelPath,device=0)


# This will give a confidence score where:
# 'Chaitanya is snoring' entails 'A man sleeps'
print("Test eval Example #1: ",
      p({'text': 'Chaitanya is snoring', 'text_pair': 'A man sleeps'}))

print("Test eval Example #2: ",
      p({'text': 'Google files for its long awaited IPO', 'text_pair': 'Google goes public'}))