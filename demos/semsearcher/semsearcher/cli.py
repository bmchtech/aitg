import os
import os
import sys
import requests
import re
import math
import itertools
import typer
import numpy as np
import json

import msgpack
import lz4.frame
from colorama import Fore, Style
from summarize_me.clean import ParagraphCleaner

DEBUG = False

app = typer.Typer()

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


# file contents
def read_file(path):
    with open(path) as f:
        return f.read()


def create_embeddings(server_uri, sentences):
    resp = requests.post(
        server_uri,
        json={
            "texts": sentences
        },
    )
    resp.raise_for_status()  # ensure

    bundle = resp.json()

    if DEBUG:
        n_embeds = bundle["num_embeds"]
        time = bundle["gen_time"]
        eprint(f"{Fore.GREEN}created ({n_embeds}) embeddings in {time:.2f}s")

    return bundle["embeds"]

def batch_list(input, size):
    it = iter(input)
    item = list(itertools.islice(it, size))
    while item:
        yield item
        item = list(itertools.islice(it, size))

@app.command("index")
def index_file(
    server: str,
    in_file: str,
    out_index: str,
    debug: bool = False,
    embed_batch_size: int = 16,
    max_sentence_length: int = 2000,
):
    server_uri = server + f"/gen_sentence_embed.json"

    if in_file == '-':
        # stdin
        contents = sys.stdin.read()
    else:
        # read full contents
        contents = read_file(in_file)

    global DEBUG
    DEBUG = debug

    # split the text into sentences
    cleaner = ParagraphCleaner()
    contents = cleaner.clean_space(contents)
    in_sentences = cleaner.sentencize(contents)
    in_sentences = list(map(lambda x: x.strip(), in_sentences))
    num_initial_sents = len(in_sentences)
    in_sentences = cleaner.drop_longer_than(in_sentences, max_sentence_length)
    num_sents = len(in_sentences)

    # embedding index
    squorgled_sentences = []

    if DEBUG:
        # print sentence overview
        eprint(f"{Fore.CYAN}split into {num_sents} sentences ({num_initial_sents} considered)")

    # embed each sentence
    in_sentenecs_batched = batch_list(in_sentences, embed_batch_size)
    for i, sent_batch in enumerate(in_sentenecs_batched):
        if DEBUG:
            eprint(
                f"{Fore.CYAN}\nembed[{(i*embed_batch_size)+1}/{num_sents}]:",
                "\n",
            )

        # embed api
        embeddings = create_embeddings(server_uri, sent_batch)

        if DEBUG:
            eprint(f"{Fore.WHITE}", end="")

        sys.stderr.flush()

        for i, embedding in enumerate(embeddings):
            # print(f"{embedding}", "\n")
            sent = sent_batch[i]

            # add entry to index
            squorgled_sentences.append([sent, embedding])

        sys.stdout.flush()

    if DEBUG:
        eprint(f"{Fore.WHITE}\ndone generating: {len(squorgled_sentences)} entries in index")
    
    # write index
    with open(out_index, 'wb') as f:
        f.write(lz4.frame.compress(msgpack.dumps(squorgled_sentences)))
        # f.write(json.dumps(squorgled_sentences).encode())


@app.command("search")
def search_index(
    server: str,
    in_index: str,
    query: str,
    n: int = 4,
    debug: bool = False
):
    server_uri = server + f"/gen_sentence_embed.json"

    # read index
    with open(in_index, 'rb') as f:
        index_data = msgpack.loads(lz4.frame.decompress(f.read()))
        # index_data = json.loads(f.read().decode())
    
    # embed query
    query_vec = create_embeddings(server_uri, [query])[0]

    # calculate similarity scores for all
    similarities = []
    for entry in index_data:
        sent_text = entry[0]
        sent_vec = entry[1]

        score = np.dot(query_vec, sent_vec)/(np.linalg.norm(query_vec)*np.linalg.norm(sent_vec))

        similarities.append([score, sent_text])

    # find top n matches
    similarities.sort(key=(lambda x: x[0]), reverse=True)

    eprint(f"{Fore.WHITE}\nshowing {n} top matches (searched {len(similarities)})")

    for i in range(0, n):
        match = similarities[i]
        score = math.floor(match[0] * 100)
        sent = match[1].strip()
        eprint(f"{Fore.GREEN}{score:2}%: {sent}\n")

def main():
    app()


if __name__ == "__main__":
    main()
