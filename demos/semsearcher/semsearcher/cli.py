import os
import time
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

from doctools.util import eprint, read_file
from semsearcher.semmine import create_document_index, search_document_index, clean_document_for_indexing

DEBUG = os.environ.get('DEBUG')

app = typer.Typer()

@app.command('clean')
def clean_file_cmd(
    in_file: str
):
    if in_file == '-':
        # stdin
        document = sys.stdin.read()
    else:
        # read full contents
        document = read_file(in_file)
    
    cleaned_sentences, num_initial_sents, num_cleaned_sents = clean_document_for_indexing(document)
    
    if DEBUG:
        # print sentence overview
        eprint(f"{Fore.CYAN}split into {num_cleaned_sents} sentences ({num_initial_sents} considered)")

        # print sentences
        for i, sentence in enumerate(cleaned_sentences):
            eprint(f"{Fore.CYAN}[{i+1}/{num_cleaned_sents}]", end='')
            print(f"{sentence}\n")

@app.command("index")
def index_file_cmd(
    server: str,
    in_file: str,
    out_index: str,
    embed_batch_size: int = 32,
    max_sentence_length: int = 2000,
):

    if in_file == '-':
        # stdin
        document = sys.stdin.read()
    else:
        # read full contents
        document = read_file(in_file)
    
    index_data = create_document_index(server, document, embed_batch_size, max_sentence_length)

    # write index
    crunched_index = lz4.frame.compress(msgpack.dumps(index_data))
    if out_index == '-':
        sys.stdout.buffer.write(crunched_index)
        sys.stdout.flush()
    else:
        # write to file
        with open(out_index, 'wb') as f:
            f.write(crunched_index)


@app.command("search")
def search_index_cmd(
    server: str,
    in_index: str,
    query: str,
    n: int = typer.Option(4, '-n', '--num-results'),
):
    # read index
    if in_index == '-':
        # stdin
        index_data = msgpack.loads(lz4.frame.decompress(sys.stdin.buffer.read()))
    else:
        # read full contents
        with open(in_index, 'rb') as f:
            index_data = msgpack.loads(lz4.frame.decompress(f.read()))

    # similarity is a tuple of (entry, score)    
    similarities = search_document_index(server, index_data, query, n)
    
    eprint(f"{Fore.WHITE}\nshowing {n} top matches (searched {len(similarities)})")

    for i in range(0, n):
        entry, score = similarities[i]
        score = math.floor(score * 100)
        sent = entry[0].strip()
        eprint(f"{Fore.GREEN}", end='')
        print(f"{score:2}%: {sent}\n")

@app.command("multisearch")
def multisearch_index_cmd(
    server: str,
    in_indexes_dir: str,
    query: str,
    n: int = typer.Option(4, '-n', '--num-results'),
):
    """
    Search all .semix files in a directory and show the top n results.
    Each result is labeled with the file name of the index it came from.
    """
    
    # build a combined index of all index files
    index_data = []
    for fn in os.listdir(in_indexes_dir):
        if fn.endswith('.semix'):
            with open(os.path.join(in_indexes_dir, fn), 'rb') as f:
                doc_name = fn.replace('.semix', '')
                indexed_sentences = msgpack.loads(lz4.frame.decompress(f.read()))
                for indexed_sentence in indexed_sentences:
                    sent = indexed_sentence[0]
                    embedding = indexed_sentence[1]
                    index_row = [sent, embedding, doc_name]
                    index_data.append(index_row)
    
    # search the combined index
    # similarity is a tuple of (entry, score)
    similarities = search_document_index(server, index_data, query, n)
    
    eprint(f"{Fore.WHITE}\nshowing {n} top matches (searched {len(similarities)})")
    
    for i in range(0, n):
        entry, score = similarities[i]
        score = math.floor(score * 100)
        sent = entry[0].strip()
        doc_name = entry[2]
        eprint(f"{Fore.GREEN}", end='')
        print(f"{score:2}%({doc_name}): {sent}\n")


def main():
    app()


if __name__ == "__main__":
    main()
