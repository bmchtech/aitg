import os
import sys
import re

from aitg_doctools.clean import ParagraphCleaner


class ArticleChunker:
    def __init__(self):
        self.cleaner = ParagraphCleaner()

    # split the text into approximate blocks ("chunks")
    def chunk(self, article, chunk_size):
        # clean article spaces
        article = self.cleaner.clean_space(article)

        # split to sentences
        sentences = self.cleaner.sentencize(article)
        # remove empty
        sentences = list(filter(None, sentences))

        # greedy recombine
        chunks = []
        current_chunk = ""
        for sent in sentences:
            # go through all sentences

            if len(current_chunk) + len(sent) <= chunk_size:
                # combine
                current_chunk += sent + " "
            else:
                # chunk full, propagate
                current_chunk = self.cleaner.clean_space(current_chunk)
                chunks.append(current_chunk)
                current_chunk = ""
                # now add this sentence to the chunk
                current_chunk += sent + " "

        # end, do final chunk
        chunks.append(self.cleaner.clean_space(current_chunk))

        return chunks
