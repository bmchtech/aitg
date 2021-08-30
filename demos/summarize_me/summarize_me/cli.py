import os
import sys
import requests
import re
import typer
from colorama import Fore, Style

from summarize_me.chunk import ArticleChunker

DEBUG = False

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

    if DEBUG:
        n_from = bundle["prompt_token_count"]
        n_to = bundle["token_count"]
        time = bundle["gen_time"]
        print(f"{Fore.GREEN}summarized ({n_from}->{n_to}) in {time:.2f}s")

    return bundle["text"]


def cli(
    server: str,
    in_file: str,
    model: str = 'bart',
    chunk_size: int = 4000,
    summary_size: int = 128,
    debug: bool = False,
):
    server_uri = server + f"/gen_{model}_summarizer.json"
    # read full contents
    contents = read_file(in_file)

    global DEBUG
    DEBUG = debug

    # chunk
    chunker = ArticleChunker()
    chunks = chunker.chunk(contents, chunk_size)

    if DEBUG:
        # print chunk overview
        print(f"{Fore.CYAN}article chunks:")
        for chunk in chunks:
            print(f"  {Fore.CYAN}chunk[{len(chunk)}]")
            # print(chunk)
            # print()
            # print()

    # summarize chunks
    for chunk in chunks:
        if DEBUG:
            print(f"{Fore.CYAN}\nsummarize:", f"{Fore.BLUE}{chunk}", "\n")

        # summarize api
        summary = summarize(server_uri, chunk, summary_size)
        # clean summarized paragraph
        summary = chunker.cleaner.clean_paragraph(summary)

        if DEBUG:
            print(f"{Fore.WHITE}", end="")

        print(f"{summary}", "\n")


def main():
    typer.run(cli)


if __name__ == "__main__":
    main()
