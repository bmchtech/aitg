import os
import sys
import requests
import re

from clean_summary import spacy_sentencize, clean_text_whitespace, clean_paragraph

server = os.environ["SERVER"]
IN_FILE = sys.argv[1]

# file contents
def read_file(path):
    with open(path) as f:
        return f.read()


# split the text into approximate blocks ("chunks")
def chunk_text(article, chunk_size):
    # clean article spaces
    article = clean_text_whitespace(article)

    # split to sentences
    sentences = spacy_sentencize(article)
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
            current_chunk = clean_text_whitespace(current_chunk)
            chunks.append(current_chunk)
            current_chunk = ""
            # now add this sentence to the chunk
            current_chunk += sent + " "

    # end, do final chunk
    chunks.append(current_chunk)

    return chunks


def summarize(article, summary_size_target):
    resp = requests.post(
        server + "/gen_bart_summarizer.json",
        json={
            "text": article,
            "num_beams": 6,
            "length_penalty": 2.0,
            "max_length": summary_size_target * 2,
            "min_length": summary_size_target,
            "no_repeat_ngram_size": 3,
        },
    )
    resp.raise_for_status()  # ensure

    bundle = resp.json()

    return bundle["text"]


# read full contents
contents = read_file(IN_FILE)

# chunk
chunks = chunk_text(contents, 1000 * 4)

# summarize chunks
for chunk in chunks:
    # print(chunk, "\n\n")
    summary = summarize(chunk, 128)
    print(summary, "\n")
