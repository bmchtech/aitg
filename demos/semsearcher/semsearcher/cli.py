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

from summarize_me.clean import ParagraphCleaner
from semsearcher.semmine import create_document_index, search_document_index
from semsearcher.util import batch_list, eprint, read_file

DEBUG = os.environ.get('DEBUG')

app = typer.Typer()

@app.command("index")
def index_file_cmd(
    server: str,
    in_file: str,
    out_index: str,
    embed_batch_size: int = 16,
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
    
    similarities = search_document_index(server, index_data, query, n)
    
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
