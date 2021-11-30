import os
import time
import sys
import requests
import re
import math
import typer
import numpy as np
import json

import msgpack
import lz4.frame
from colorama import Fore, Style
from doctools.clean import ParagraphCleaner
from doctools.util import batch_list, eprint

DEBUG = os.environ.get('DEBUG')

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

def create_document_index(
    server: str,
    contents: str,
    embed_batch_size: int = 16,
    max_sentence_length: int = 2000,
):
    server_uri = server + f"/gen_sentence_embed.json"
    start_time = time.time()

    # split the text into sentences
    cleaner = ParagraphCleaner()
    contents = cleaner.clean_space(contents)
    in_sentences = cleaner.sentencize(contents)
    in_sentences = list(map(lambda x: x.strip(), in_sentences))
    num_initial_sents = len(in_sentences)
    in_sentences = cleaner.drop_longer_than(in_sentences, max_sentence_length)
    num_sents = len(in_sentences)

    # embedding index
    index_data = []

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
            index_data.append([sent, embedding])

        sys.stdout.flush()

    if DEBUG:
        eprint(f"{Fore.WHITE}\ndone generating in {time.time() - start_time:.2f}s: {len(index_data)} entries in index")
    
    return index_data

def search_document_index(
    server: str,
    index_data,
    query: str,
    n,
):
    server_uri = server + f"/gen_sentence_embed.json"
    
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

    return similarities