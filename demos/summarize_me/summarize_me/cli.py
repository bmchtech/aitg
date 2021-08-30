import os
import sys
import requests
import re

from summarize_me.chunk import ArticleChunker

# file contents
def read_file(path):
    with open(path) as f:
        return f.read()


def summarize(server_uri, article, summary_size_target):
    resp = requests.post(
        server_uri,
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


def main():
    server = os.environ["SERVER"]
    in_file = sys.argv[1]

    server_uri = server + "/gen_bart_summarizer.json"
    # read full contents
    contents = read_file(in_file)

    # chunk
    chunker = ArticleChunker()
    chunks = chunker.chunk(contents, 1000 * 4)

    # summarize chunks
    for chunk in chunks:
        # summarize api
        summary = summarize(server_uri, chunk, 128)
        # clean summarized paragraph
        summary = chunker.cleaner.clean_paragraph(summary)
        print(summary, "\n")

if __name__ == "__main__":
    main()
