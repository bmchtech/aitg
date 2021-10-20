import os
import sys
import requests
import re
import typer
from colorama import Fore, Style

from doctools.util import read_file
from summarize_me.summgen import summarize_document

DEBUG = os.environ.get('DEBUG')

def cli(
    server: str,
    in_file: str,
    model: str = "bart",
    chunk_size: int = 4000,
    summary_size_min: int = 128, # recommend 128 or 256
    summary_size_max: int = 256,
):
    if in_file == '-':
        # stdin
        document = sys.stdin.read()
    else:
        # read full contents
        document = read_file(in_file)

    document_summary = summarize_document(server, document, model, chunk_size, summary_size_min, summary_size_max)

    if DEBUG:
        ratio = (len(document_summary) / len(document)) * 100
        eprint(f"{Fore.WHITE}\nfull summary ({ratio:.0f}%):\n")
    print(document_summary)

def main():
    typer.run(cli)


if __name__ == "__main__":
    main()
