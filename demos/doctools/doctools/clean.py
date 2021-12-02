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
        # spacy for nlp senter
        spacy_model = "en_core_web_sm"
        self.nlp_senter = spacy.load(spacy_model, exclude=["parser"])
        self.nlp_senter.enable_pipe("senter")
        self.nlp_senter.max_length = 100000000 # length isn't really too badly when using sentence detection

        # spacy for nlp tagger
        self.nlp_tagger = spacy.load(spacy_model, exclude=["parser", "ner"])
        self.nlp_tagger.add_pipe('sentencizer')

    def sentencize(self, paragraph):
        """
        Sentencize the paragraph using spacy
        """
        spacy_doc = self.nlp_senter(paragraph)
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
        """
        Clean up the text by removing extra spaces and newlines
        """
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
    
    def sentence_is_paragraph_form(self, sentence):
        """
        Check if the sentence is a paragraph form
        """
        doc = self.nlp_tagger(sentence)
        sent = list(doc.sents)[0]

        # check this sent content
        nouns = 0
        verbs = 0
        if sent[-1].is_punct: # make sure sentence ends with punctuation
            for token in sent:
                if token.pos_ in ["NOUN", "PROPN", "PRON"]:
                    nouns += 1
                elif token.pos_ in ["VERB", "AUX"]:
                    verbs += 1

        # so we want to make sure this is a complete sentence
        if nouns > 0 and verbs > 0:
            return True

        return False
    
    def filter_non_paragraph_sentences(self, sentences):
        """
        Filter out non-paragraph sentences
        """
        accepted = []
        rejected = []
        for sent in sentences:
            if self.sentence_is_paragraph_form(sent):
                accepted.append(sent)
            else:
                rejected.append(sent)

        return accepted, rejected
