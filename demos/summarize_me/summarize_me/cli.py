import os
import sys
import requests
import re
import typer
from colorama import Fore, Style
from types import SimpleNamespace

from aitg_doctools.util import read_file, eprint
from summarize_me.summgen import summarize_document

DEBUG = os.environ.get('DEBUG')

def cli(
    server: str,
    in_file: str,
    headline: str = None,
    model: str = "bart",
    chunk_size: int = 4000,
    summary_size_min: int = 16,
    summary_size_max: int = 256,
    
    typical_p: float = None,
    num_beams: int = None,
    no_repeat_ngram_size: int = None,
    early_stopping: bool = False,
):
    if in_file == '-':
        # stdin
        document = sys.stdin.read()
    else:
        # read full contents
        document = read_file(in_file)
    
    # create generation params
    gen_params = SimpleNamespace(
        typical_p=typical_p,
        num_beams=num_beams,
        no_repeat_ngram_size=no_repeat_ngram_size,
        early_stopping=early_stopping,
    )

    document_summary = summarize_document(server, document, headline, model, chunk_size, summary_size_min, summary_size_max, gen_params)

    if DEBUG:
        ratio = (len(document_summary) / len(document)) * 100
        eprint(f"{Fore.WHITE}\nfull summary ({ratio:.0f}%):\n")
    print(document_summary)

def main():
    typer.run(cli)


if __name__ == "__main__":
    main()
