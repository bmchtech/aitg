import os
import sys
import json
import re

from sacremoses import MosesDetokenizer
import spacy

with open(sys.argv[1]) as f:
    source_json = json.load(f)

def filter_text(text):
    # clean whitespace
    text = text.strip()
    text = re.sub("\s\s+", " ", text)
    return text

def filter_token(tok):
    tok = tok.replace('Ġ', '')
    return tok

# load tokens
source_text = filter_text(source_json["text"])
source_tokens = source_text.split(' ')
# print(source_tokens)

# filter tokens
filtered_tokens = [filter_token(token) for token in source_tokens]

# detokenize to string
md = MosesDetokenizer(lang='en')
detok_str = md.detokenize(filtered_tokens)

# truecase the text with spacy
nlp = spacy.load("en_core_web_sm", exclude=["parser"])
nlp.enable_pipe("senter")
spacy_doc = nlp(detok_str)

processed_sents = []
for sent in spacy_doc.sents:
    # process
    frag = sent.text
    frag = frag.capitalize() # upcase
    processed_sents.append(frag)

result = ' '.join(processed_sents)
print(result)