import spacy

# Load the pre-trained English model
# Execute this before: python -m spacy download en_core_web_sm
nlp = spacy.load('en_core_web_sm')
# Sample text for NER
text = "Give me more information on user Robert Downey Jr. and the company Apple Inc. " \
       "in the city of San Francisco, California. " 

# Process the text with the spaCy model
doc = nlp(text)

# Iterate over the detected entities
for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)

'''
PERSON: Names of people, including fictional characters.
ORG: Companies, agencies, institutions, etc.
GPE: Countries, cities, and states.
LOC: Non-GPE locations like mountain ranges or bodies of water.
DATE: Absolute or relative dates and periods.
TIME: Times smaller than a day.
MONEY: Monetary values.
PRODUCT: Objects, vehicles, foods.
EVENT: Named events like hurricanes or sports events.
WORK_OF_ART: Titles of books, songs, etc.
LAW: Named documents made into laws.
LANGUAGE: Named languages.
PERCENT: Percentages.
QUANTITY: Measurements.
ORDINAL: "first", "second", etc.
CARDINAL: Numerals not covered by other types.
NORP: Nationalities or religious or political groups.
'''
