import os
import sys
import json
import re

from sacremoses import MosesDetokenizer
import spacy


class ParagraphCleaner:
    def __init__(self):
        # init moses
        self.detokenizer = MosesDetokenizer(lang="en")
        # init spacy
        self.nlp = spacy.load("en_core_web_sm", exclude=["parser"])
        self.nlp.enable_pipe("senter")
        self.nlp.max_length = 100000000 # length isn't really too badly when using sentence detection

    def sentencize(self, paragraph):
        spacy_doc = self.nlp(paragraph)
        return [sent.text for sent in spacy_doc.sents]

    def truecase(self, paragraph):
        # truecase the text with spacy
        sentences = self.sentencize(paragraph)

        processed_sents = []
        for sent in sentences:
            # process
            frag = sent
            frag = frag.capitalize()  # upcase
            processed_sents.append(frag)

        result = " ".join(processed_sents)
        return result

    def clean_space(self, text):
        # clean whitespace
        text = text.replace("\n", " ")
        text = text.strip()
        text = re.sub("\s\s+", " ", text)
        return text

    def clean_paragraph(self, source_text):
        def clean_token(tok):
            tok = tok.replace("Ġ", "")
            return tok

        # prefilter text
        source_text = self.clean_space(source_text)

        # split into tokens rudimentarily
        source_tokens = source_text.split(" ")

        # filter tokens
        filtered_tokens = [clean_token(token) for token in source_tokens]

        # detokenize to string with moses
        detokenized_str = self.detokenizer.detokenize(filtered_tokens)

        result = self.truecase(detokenized_str)

        return result
    
    def drop_longer_than(self, sentences, drop_length):
        results = []
        for sent in sentences:
            if len(sent) <= drop_length:
                results.append(sent)
        return results
