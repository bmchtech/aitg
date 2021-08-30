import os
import sys
import requests
import re

from chunk import ArticleChunker

server = os.environ["SERVER"]
IN_FILE = sys.argv[1]

# file contents
def read_file(path):
    with open(path) as f:
        return f.read()


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
chunker = ArticleChunker()
chunks = chunker.chunk(contents, 1000 * 4)

# summarize chunks
for chunk in chunks:
    # summarize api
    summary = summarize(chunk, 128)
    # clean summarized paragraph
    summary = chunker.cleaner.clean_paragraph(summary)
    print(summary, "\n")
