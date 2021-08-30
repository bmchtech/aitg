import os
import sys
import requests
import re
import typer
from colorama import Fore, Style

from summarize_me.chunk import ArticleChunker

DEBUG = False


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


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
            "max_length": min(1024, summary_size_target * 2),
            "min_length": summary_size_target,
            "no_repeat_ngram_size": 4,
        },
    )
    resp.raise_for_status()  # ensure

    bundle = resp.json()

    if DEBUG:
        n_from = bundle["prompt_token_count"]
        n_to = bundle["token_count"]
        time = bundle["gen_time"]
        eprint(f"{Fore.GREEN}summarized ({n_from}->{n_to}) in {time:.2f}s")

    return bundle["text"]


def cli(
    server: str,
    in_file: str,
    model: str = "bart",
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
        eprint(f"{Fore.CYAN}article chunks:")
        for i, chunk in enumerate(chunks):
            eprint(f"  {Fore.CYAN}chunk #{i}: [{len(chunk)}]")
            # print(chunk)
            # print()
            # print()

    # summarize chunks
    num_chunks = len(chunks)
    for i, chunk in enumerate(chunks):
        if DEBUG:
            eprint(
                f"{Fore.CYAN}\nsummarize[{i+1}/{num_chunks}]:",
                f"{Fore.BLUE}{chunk}",
                "\n",
            )

        # summarize api
        summary = summarize(server_uri, chunk, summary_size)
        # clean summarized paragraph
        summary = chunker.cleaner.clean_paragraph(summary)

        if DEBUG:
            eprint(f"{Fore.WHITE}", end="")

        sys.stderr.flush()

        print(f"{summary}", "\n")

        sys.stdout.flush()


def main():
    typer.run(cli)


if __name__ == "__main__":
    main()
