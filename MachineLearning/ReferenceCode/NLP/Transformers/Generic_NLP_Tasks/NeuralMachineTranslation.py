# pip install transfomers
# pip install sentencepiece transformers[sentencepiece]

# sentence piece is required by the translation model, else you will get error.

# NeuralMachineTranslation (NMT) is translation of one text to another.
#
# Source Data has been downloaded from:
# http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip
#
# The above ZIP contains spa.txt that has one-on-one translation of words 
# and simple sentences from English to Spanish

# BLEU score is used to measure the accuracy of a translation.
# BLEU (BiLingual Evaluation Understudy) is a metric for automatically 
# evaluating machine-translated text. The BLEU score is a number between 
# zero and one that measures the similarity of the machine-translated 
# text to a set of high quality reference translations.

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import RegexpTokenizer # Tokenize/Split words baed on RegEx
from transformers import pipeline

''' Sample Data
Run.	Corred.
Who?	¿Quién?
Fire!	¡Fuego!
Fire!	¡Incendio!
Fire!	¡Disparad!
Help!	¡Ayuda!
Help!	¡Socorro! ¡Auxilio!
Help!	¡Auxilio!
'''

# Store a map of each english word to its spanish equivalent
eng2spa = {}
for line in open('./data/spa.txt',encoding='utf8'): # Encoding is needed as file is not in right encoding
  line = line.rstrip()
  eng, spa = line.split("\t") # English and Spanish words are split by tab
  if eng not in eng2spa:
    eng2spa[eng] = []
  eng2spa[eng].append(spa)

# Match 1 or more alphanumeric Char. 
# Using this tokenizer with the given RegEx will remove the punctuations
tokenizer = RegexpTokenizer(r'\w+') 

# Test
tokens = tokenizer.tokenize('Estamos en casa.'.lower()) 
print(tokens)
print([tokens])
# 1st arguement to list is set of acceptable translations, which are computed manually 
# 2nd argument is the hypothesis or the model output. 
# So where we are comparing all the possible manually generated translationa
# with the since output given by the model
# sentence_bleu analyzes upto 4-gram similarities and assigns 
# equal weights of 0.25 to each. 
print("BLEU Score:",sentence_bleu([tokens],tokens))

# The above comparison gives very low score even though the text is same, so should 
# have a score of 1.0. This is because the accuracy is being compared with upto 4-grams
# even though what we passed were unigram. Hence we need to adjust the weights so that 
# all the weightage is given to unigrams.
# The following code shows this adjustment, where unigrams are assigned a weight of 1.0, 
# while 2,3 and 4-grams have weight = 0. This will give the correct BLEW score of 1.0
print("BLEU Score with weights:",sentence_bleu([tokens],tokens, weights=[1.0, 0, 0, 0]))


# When using less than 4 words, we get a warning to use SmoothingFunction 
# if weights are not assigned. Let use use a smoothing 
smoother = SmoothingFunction()
print("BLEU Score with weights:",sentence_bleu([tokens],tokens, weights=[1.0, 0, 0, 0]))

# Get the tokens for the spanish translations
eng2spa_tokens = {}
for eng, spa_list in eng2spa.items():
  spa_list_tokens = []
  for text in spa_list: # For each Spanish translation
    tokens = tokenizer.tokenize(text.lower()) # tokens will be a list
    spa_list_tokens.append(tokens)
  eng2spa_tokens[eng] = spa_list_tokens # list of list of tokens

# The model specification contains source and target language
translator = pipeline("translation",
                      model='Helsinki-NLP/opus-mt-en-es') #, device=0) No GPU is available

# Sample translation
print("Sample Translation:",translator("My name is Chaitanya Belwal"))

# Now let's convert text from the test corpus.
engPhrases = list(eng2spa.keys())
print("Total number of entries:",len(engPhrases))

# Let's only take 10 entries
# This will execute slowly in CPUs, only increase count if GPU is available
engPhraseSel = engPhrases[1000:1010]

# Not call the model and get a testTranslations Vector:
testTranslations = translator(engPhraseSel)

scores = []
scores_weights = []
for eng, pred in zip(engPhraseSel, testTranslations):
  actualTranslation = eng2spa_tokens[eng]

  # tokenize the predicted translations
  # Need to tokenize it as sentence_bleu accepts tokenized text only
  spa_pred = tokenizer.tokenize(pred['translation_text'].lower())

  score_weight = sentence_bleu(actualTranslation, spa_pred,weights=[1.0, 0, 0, 0])
  score = sentence_bleu(actualTranslation, spa_pred)

  scores_weights.append(score_weight)
  scores.append(score)

print("All Scores, without n-gram weights:",scores)
print("All Scores, with n-gram weights:",scores_weights)