import os
import sys
import requests
import re
import typer
from colorama import Fore, Style

from aitg_doctools.util import eprint
from aitg_doctools.chunk import ArticleChunker

DEBUG = os.environ.get("DEBUG")
KEY = os.environ.get("KEY") or ""


def summarize_local_v1(
    server_uri, article, summary_size_min, summary_size_max, gen_params
):
    req_bundle = {
        "key": KEY,
        "text": article,
        "max_length": min(1024, summary_size_max),
        "min_length": summary_size_min,
    }

    # add gen params
    for k, v in gen_params.__dict__.items():
        if v is not None:
            req_bundle[k] = v
    if DEBUG:
        eprint(f"{Fore.CYAN}requesting {server_uri}, min={summary_size_min}, max={summary_size_max}, gen_params={gen_params}")

    resp = requests.post(
        server_uri,
        json=req_bundle,
    )
    resp.raise_for_status()  # ensure

    bundle = resp.json()

    if DEBUG:
        n_from = bundle["prompt_token_count"]
        n_to = bundle["token_count"]
        time = bundle["gen_time"]
        eprint(f"{Fore.GREEN}summarized ({n_from}->{n_to}) in {time:.2f}s")

    return bundle["text"]

def summarize_openai_v2(article, headline):
    # create prompt
    openai_prompt = 'Summarize this chunk of text from {}. Do not reference the text with words like "this text", but reference the author when relevant to explaining their point of view. The goal is to simply create an information-dense, concise version of the input text.'.format(headline)
    openai_prompt += "\n"
    openai_prompt += "Input:\n"
    openai_prompt += article
    openai_prompt += "\n"
    openai_prompt += "Output:\n"

    if DEBUG:
        eprint(f"{Fore.GREEN}openai prompt:\n{openai_prompt}\n")

    import openai

    # summarize
    openai.api_key = os.getenv("OPENAI_API_KEY")

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=openai_prompt,
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    # extract summary
    summary = response["choices"][0]["text"]

    if DEBUG:
        eprint(f"{Fore.GREEN}openai summary:\n{summary}\n")

    return summary

def summarize_document(
    server: str,
    document: str,
    headline: str,
    model,
    chunk_size,
    summary_size_min,
    summary_size_max,
    gen_params,
):
    server_uri = server + f"/gen_{model}_summarizer.json"

    # chunk
    chunker = ArticleChunker()
    chunks = chunker.chunk(document, chunk_size)

    # # edit chunks, add headlines
    # for i, chunk in enumerate(chunks):
    #     if headline:
    #         # prepend headline to each chunk
    #         chunks[i] = headline + "\n\n" + chunk

    if DEBUG:
        # print chunk overview
        eprint(f"{Fore.CYAN}article chunks:")
        for i, chunk in enumerate(chunks):
            eprint(f"  {Fore.CYAN}chunk #{i}: [{len(chunk)}]")

    # summarize chunks
    num_chunks = len(chunks)
    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        # for all chunks
        if DEBUG:
            eprint(
                f"{Fore.CYAN}\nsummarize[{i+1}/{num_chunks}]:",
                f"{Fore.BLUE}{chunk}",
                "\n",
            )

        # summarize api
        summary = summarize_openai_v2(chunk, headline)
        # clean summarized paragraph
        summary = chunker.cleaner.clean_paragraph(summary)

        if DEBUG:
            eprint(f"{Fore.WHITE}", end="")
            eprint(f"{summary}", "\n")
            sys.stderr.flush()

        # store summary chunk
        chunk_summaries.append(summary)

    combined_summaries = "\n\n".join(chunk_summaries)
    return combined_summaries
