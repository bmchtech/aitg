import time
import os
import sys
from typing import Optional
from math import floor
import typer
import colorama
from colorama import Fore, Back, Style
from aitg import __version__, ICON_ART


def version_callback(value: bool):
    if value:
        typer.echo(f"{__version__}")
        raise typer.Exit()

def info_callback(value: bool):
    if value:
        from aitg.util import get_compute_device
        import torch
        device_info = get_compute_device()
        pad = '            '
        banner = f'AITG HOST v{__version__}'
        underline = len(banner) * '‾'
        print(ICON_ART, f'\n{pad}{banner}')
        print(f'{pad}{underline}')
        print(f'{pad} ⊦ PLATFORM: {sys.platform}')
        print(f'{pad} ⊦ DEVICE: {device_info[0]}')
        print(f'{pad}   ⊦ MEMORY: {device_info[2]:.2f} GB')
        raise typer.Exit()

def cli(
    version: Optional[bool] = typer.Option(
        None, "-v", "--version", callback=version_callback, is_eager=True
    ),
    info: Optional[bool] = typer.Option(
        None, "--info", callback=info_callback, is_eager=True
    ),
    temp: float = 0.9,
    max_length: int = 256,
    min_length: int = 0,
    context_amount: float = 0.5,
    seed: int = None,
    typical_p: float = 0.9,
    repetition_penalty: float = 1.0,
    length_penalty: float = 1.0,
    max_time: float = None,
    no_repeat_ngram_size: int = 0,
    reuse_session: bool = False,
):
    colorama.init()

    model_dir = os.environ.get("MODEL")
    if not model_dir:
        raise RuntimeError("no model specified. please pass a path to your model in the MODEL environment variable")

    start = time.time()
    print(Style.NORMAL + Fore.CYAN + f"initializing", end="")

    # imports here, because they're slow
    from aitg.model import load_gpt_model
    from aitg.util import multiline_in, get_compute_device
    from aitg.gens.sliding_generator import SlidingGenerator

    print(f"[{get_compute_device()[1]}]...")

    print(Style.DIM + Fore.RESET + f"[dbg] init in: {time.time() - start:.2f}s")

    start = time.time()
    print(Style.NORMAL + Fore.CYAN + "loading model...")
    ai = load_gpt_model(model_dir)
    print(
        Style.DIM
        + Fore.RESET
        + f"[dbg] finished loading in: {time.time() - start:.2f}s"
    )
    print(Style.DIM + Fore.RESET + f"[dbg] model: {ai.model_name} ({ai.model_type})")

    # prompt
    slidegen = SlidingGenerator(ai)
    while True:
        print(Style.NORMAL + Fore.WHITE + "\nprompt" + Fore.GREEN + ":")
        prompt = multiline_in()
        gen_type = "generating"
        is_fresh = True

        # if reuse enabled, go not fresh by default
        if reuse_session:
            is_fresh = False

        if prompt == "" and len(slidegen.token_log) > 0:
            gen_type = "continuing"
            is_fresh = False
            context, context_toks = slidegen.next_context(max_length, context_amount)
            print(
                Style.DIM + Fore.RESET + f"context[{len(context_toks)}]: {context}",
                end="",
            )

        print(Style.NORMAL + Fore.WHITE + "□\n――――――――――")
        print(Style.DIM + Fore.RESET + f"{gen_type}...", end="")

        start = time.time()

        output = slidegen.generate(
            prompt=prompt,
            fresh=is_fresh,
            context_amount=context_amount,
            min_length=min_length,
            max_length=max_length,
            seed=seed,
            temperature=temp,
            typical_p=typical_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            max_time=max_time,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )

        generation_time = time.time() - start
        total_gen_num = len(output.tokens)
        print(
            Style.DIM
            + Fore.RESET
            + f"[{output.num_new}/{total_gen_num}] ({generation_time:.2f}s/{(output.num_new/generation_time):.2f}tps)"
        )
        # print(Style.NORMAL + Fore.MAGENTA + f"{output.tokens}")
        if is_fresh:
            print(
                Style.NORMAL + Fore.MAGENTA + f"{ai.filter_text(output.text)}", end=""
            )
        else:
            print(
                Style.NORMAL
                + Fore.MAGENTA
                + f"{ai.filter_text(slidegen.toks_to_str(slidegen.token_log))}",
                end="",
            )
        print(Style.DIM + Fore.RESET + "□")
        if output.num_new == 0:
            # no more tokens
            print(Style.DIM + Fore.RED + "□ □ □")
        print("\n")


def main():
    typer.run(cli)

if __name__ == "__main__":
    main()
