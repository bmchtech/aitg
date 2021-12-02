import sys

import spacy

spacy_model = "en_core_web_sm"
nlp = spacy.load(spacy_model, exclude=["parser", "ner"])
nlp.add_pipe('sentencizer')

sentence = sys.argv[1]

doc = nlp(sentence)
sent = list(doc.sents)[0]

# check this sent content
nouns = 0
verbs = 0
complex = 0
punct = 0
words = 0

end_punct = sent[-1].is_punct
for token in sent:
    words += 1
    if token.pos_ in ["NOUN", "PROPN", "PRON"]:
        nouns += 1
    elif token.pos_ in ["VERB", "AUX"]:
        verbs += 1
    elif token.pos_ in ["ADV", "ADP", "PART"]:
        complex += 1
    elif token.pos_ in ["PUNCT"]:
        punct += 1

# # so we want to make sure this is a complete sentence

# log nouns and verbs
for token in doc:
    print(token.text, token.lemma_, token.pos_, token.tag_)
# print(f"{nouns} nouns, {verbs} verbs")
print(f"nouns: {nouns}, verbs: {verbs}")
print(f"end punctuation: {end_punct}")

if nouns > 0 and verbs > 0:
    print('accept: nouns and verbs')

if (punct >= 0.3 * words):
    print('reject: excessive punctuation')

if nouns > 2 and complex > 0:
    print('accept: many nouns')

if complex > 2:
    print('accept: complex pos')