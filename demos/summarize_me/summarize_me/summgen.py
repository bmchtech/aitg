import os
import sys
import requests
import re
import typer
from colorama import Fore, Style

from aitg_doctools.util import eprint
from aitg_doctools.chunk import ArticleChunker

DEBUG = os.environ.get('DEBUG')
KEY = os.environ.get('KEY') or ''

def summarize(server_uri, article, summary_size_min, summary_size_max, typical_p):
    resp = requests.post(
        server_uri,
        json={
            "key": KEY,
            "text": article,
            "max_length": min(1024, summary_size_max),
            "min_length": summary_size_min,
            "typical_p": typical_p,
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

def summarize_document(
    server: str,
    document: str,
    model,
    chunk_size,
    summary_size_min,
    summary_size_max,
    typical_p,
):
    server_uri = server + f"/gen_{model}_summarizer.json"

    # chunk
    chunker = ArticleChunker()
    chunks = chunker.chunk(document, chunk_size)

    if DEBUG:
        # print chunk overview
        eprint(f"{Fore.CYAN}article chunks:")
        for i, chunk in enumerate(chunks):
            eprint(f"  {Fore.CYAN}chunk #{i}: [{len(chunk)}]")

    # summarize chunks
    num_chunks = len(chunks)
    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        if DEBUG:
            eprint(
                f"{Fore.CYAN}\nsummarize[{i+1}/{num_chunks}]:",
                f"{Fore.BLUE}{chunk}",
                "\n",
            )

        # summarize api
        summary = summarize(server_uri, chunk, summary_size_min, summary_size_max, typical_p)
        # clean summarized paragraph
        summary = chunker.cleaner.clean_paragraph(summary)

        if DEBUG:
            eprint(f"{Fore.WHITE}", end="")
            eprint(f"{summary}", "\n")
            sys.stderr.flush()

        # store summary chunk
        chunk_summaries.append(summary)
    
    combined_summaries = '\n\n'.join(chunk_summaries)
    return combined_summaries