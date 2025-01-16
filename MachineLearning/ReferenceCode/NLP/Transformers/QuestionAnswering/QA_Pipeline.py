# pip install transfomers

# This pipeline deals with question answering.
# this uses the question-answering pipeline
# the context contains the answer

from transformers import pipeline

# variable for pipeline shows what it does
qa_pipeline = pipeline("question-answering")

context = "My favorite food is goat biryani"
question = "What is my favorite food?"

print("Sample run #1:",qa_pipeline(context=context, question=question))

context = "Albert Einstein (14 March 1879 – 18 April 1955) was a " + \
  "German-born theoretical physicist, widely acknowledged to be one of the " + \
  "greatest physicists of all time. Einstein is best known for developing " + \
  "the theory of relativity, but he also made important contributions to " + \
  "the development of the theory of quantum mechanics. Relativity and " + \
  "quantum mechanics are together the two pillars of modern physics."

question = "When was Albert Einstein born?"

# Note that in the context we do not explicitly call out the birth date
# but the model understands natural language.
print("Sample run #2:",qa_pipeline(context=context, question=question))
