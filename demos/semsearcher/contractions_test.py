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
        {ORTH: "do", LEMMA: "do"},
        {ORTH: "n't", LEMMA: "not", NORM: "not", TAG: "RB"}],
    "doesn't": [
        {ORTH: "does", LEMMA: "do"},
        {ORTH: "n't", LEMMA: "not", NORM: "not", TAG: "RB"}],
    "didn't": [
        {ORTH: "did", LEMMA: "do"},
        {ORTH: "n't", LEMMA: "not", NORM: "not", TAG: "RB"}],
# can
    "can't": [
        {ORTH: "ca", LEMMA: "can"},
        {ORTH: "n't", LEMMA: "not", NORM: "not", TAG: "RB"}],
    "couldn't": [
        {ORTH: "could", LEMMA: "can"},
        {ORTH: "n't", LEMMA: "not", NORM: "not", TAG: "RB"}],
# have
    "I've'": [
        {ORTH: "I", LEMMA: "I"},
        {ORTH: "'ve'", LEMMA: "have", NORM: "have", TAG: "VERB"}],
    "haven't": [
        {ORTH: "have", LEMMA: "have"},
        {ORTH: "n't", LEMMA: "not", NORM: "not", TAG: "RB"}],
    "hasn't": [
        {ORTH: "has", LEMMA: "have"},
        {ORTH: "n't", LEMMA: "not", NORM: "not", TAG: "RB"}],
    "hadn't": [
        {ORTH: "had", LEMMA: "have"},
        {ORTH: "n't", LEMMA: "not", NORM: "not", TAG: "RB"}],
# will/shall will be replaced by will
    "I'll'": [
        {ORTH: "I", LEMMA: "I"},
        {ORTH: "'ll'", LEMMA: "will", NORM: "will", TAG: "VERB"}],
    "he'll'": [
        {ORTH: "he", LEMMA: "he"},
        {ORTH: "'ll'", LEMMA: "will", NORM: "will", TAG: "VERB"}],
    "she'll'": [
        {ORTH: "she", LEMMA: "she"},
        {ORTH: "'ll'", LEMMA: "will", NORM: "will", TAG: "VERB"}],
    "it'll'": [
        {ORTH: "it", LEMMA: "it"},
        {ORTH: "'ll'", LEMMA: "will", NORM: "will", TAG: "VERB"}],
    "won't": [
        {ORTH: "wo", LEMMA: "will"},
        {ORTH: "n't", LEMMA: "not", NORM: "not", TAG: "RB"}],
    "wouldn't": [
        {ORTH: "would", LEMMA: "will"},
        {ORTH: "n't", LEMMA: "not", NORM: "not", TAG: "RB"}],
# be
    "I'm'": [
        {ORTH: "I", LEMMA: "I"},
        {ORTH: "'m'", LEMMA: "be", NORM: "am", TAG: "VERB"}]
}

# add special cases
for word in contraction_tokenizer_exceptions.keys():
    nlp.tokenizer.add_special_case(word, contraction_tokenizer_exceptions[word])

#testing all contractions using spaCy's update tokenizer
doc1 = nlp(u"Oh no he didn't. I can't and I won't. I'll know what I'm gonna do.")
for token in doc1:
    print(token.text, token.lemma_, token.pos_)
