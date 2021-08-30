import os
import sys
import json
import re

from sacremoses import MosesDetokenizer
import spacy

def spacy_sentencize(paragraph):
    nlp = spacy.load("en_core_web_sm", exclude=["parser"])
    nlp.enable_pipe("senter")
    spacy_doc = nlp(paragraph)
    return spacy_doc.sents

def spacy_truecase(paragraph):
    # truecase the text with spacy

    sentences = spacy_sentencize(paragraph)

    processed_sents = []
    for sent in sentences:
        # process
        frag = sent.text
        frag = frag.capitalize() # upcase
        processed_sents.append(frag)

    result = ' '.join(processed_sents)
    return result

def clean_text_whitespace(text):
    # clean whitespace
    text = text.strip()
    text = re.sub("\s\s+", " ", text)
    return text

def clean_paragraph(source_text):
    def clean_token(tok):
        tok = tok.replace('Ġ', '')
        return tok

    # prefilter text
    source_text = clean_text_whitespace(source_text)

    # split into tokens rudimentarily
    source_tokens = source_text.split(' ')

    # filter tokens
    filtered_tokens = [clean_token(token) for token in source_tokens]

    # detokenize to string with moses
    md = MosesDetokenizer(lang='en')
    detok_str = md.detokenize(filtered_tokens)

    result = spacy_truecase(detok_str)

    return result

if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        source_json = json.load(f)
    result = clean_paragraph(source_json["text"])
    print(result)