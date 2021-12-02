from __future__ import unicode_literals, print_function
import spacy
from spacy.attrs import ORTH, LEMMA, NORM, TAG
from spacy.lang.en.stop_words import STOP_WORDS
from spacy import displacy
from pathlib import Path
import re

nlp = spacy.load('en_core_web_sm')
# example of expanding contractions using regexes (slow for a big corpus)
text_with_contractions = "Oh no he didn't."
# text_without_contractions = re.sub(r'(\w+)n\'t', r'\g<1>' + " not", text_with_contractions)
# print(text_without_contractions)

'''
dealing with contractions by expanding spaCy's tokenizer exceptions
ORTH is the form in the text/corpus
LEMMA is the dictionary form
TAG is part of speech
'''
contraction_tokenizer_exceptions = {
# do
    "don't": [
        {ORTH: "do"},
        {ORTH: "n't", NORM: "not",}],
    "doesn't": [
        {ORTH: "does"},
        {ORTH: "n't", NORM: "not",}],
    "didn't": [
        {ORTH: "did"},
        {ORTH: "n't", NORM: "not",}],
# can
    "can't": [
        {ORTH: "ca"},
        {ORTH: "n't", NORM: "not",}],
    "couldn't": [
        {ORTH: "could"},
        {ORTH: "n't", NORM: "not",}],
# have
    "I've'": [
        {ORTH: "I"},
        {ORTH: "'ve'", NORM: "have",}],
    "haven't": [
        {ORTH: "have"},
        {ORTH: "n't", NORM: "not",}],
    "hasn't": [
        {ORTH: "has"},
        {ORTH: "n't", NORM: "not",}],
    "hadn't": [
        {ORTH: "had"},
        {ORTH: "n't", NORM: "not",}],
# will/shall will be replaced by will
    "I'll'": [
        {ORTH: "I"},
        {ORTH: "'ll'", NORM: "will",}],
    "he'll'": [
        {ORTH: "he"},
        {ORTH: "'ll'", NORM: "will",}],
    "she'll'": [
        {ORTH: "she"},
        {ORTH: "'ll'", NORM: "will",}],
    "it'll'": [
        {ORTH: "it"},
        {ORTH: "'ll'", NORM: "will",}],
    "won't": [
        {ORTH: "wo"},
        {ORTH: "n't", NORM: "not",}],
    "wouldn't": [
        {ORTH: "would"},
        {ORTH: "n't", NORM: "not",}],
# be
    "I'm'": [
        {ORTH: "I"},
        {ORTH: "'m'", NORM: "am",}]
}

# add special cases
# for word in contraction_tokenizer_exceptions.keys():
    # nlp.tokenizer.add_special_case(word, contraction_tokenizer_exceptions[word])

#testing all contractions using spaCy's update tokenizer
doc1 = nlp(u"Oh no he didn't. I can't and I won't. I'll know what I'm gonna do.")
for token in doc1:
    print(token.text, token.lemma_, token.pos_)
