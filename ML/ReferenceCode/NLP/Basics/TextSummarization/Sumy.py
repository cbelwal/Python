#pip install sumy
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
import textwrap as textwrap
import pandas as pd


#Dataset from: https://www.kaggle.com/shivamkushwaha/bbc-full-text-document-classification
input_file= 'c:/users/chaitanya belwal/.datasets/nlp/bbc_text_cls.csv'

df = pd.read_csv(input_file)
labels = set(df['labels'])
label = 'business'


texts = df[df['labels'] == label]['text']
article = textwrap.fill(texts[0],fix_sentence_endings=True)

#-- Create parser object
parser = PlaintextParser.from_string(
    article.split("\n", 1)[1],
    Tokenizer("english"))

#--- TextRank
print("Summary on textRank:")
summarizer = TextRankSummarizer()
summary = summarizer(parser.document, sentences_count=5)
#Wrap it for easier reading
for s in summary:
  print(textwrap.fill(str(s),fix_sentence_endings=True))

#--- LSA Summarizer
print("Summary on LSA:")
summarizer = LsaSummarizer()
summary = summarizer(parser.document, sentences_count=5)
for s in summary:
  print(textwrap.fill(str(s),fix_sentence_endings=True))


#Can also use Gensim library