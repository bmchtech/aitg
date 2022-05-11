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
    typical_p: float = 0.9,
    max_time: float = None,
):
    colorama.init()

    model_dir = os.environ.get("MODEL")
    if not model_dir:
        raise RuntimeError("no model specified. please pass a path to your model in the MODEL environment variable")

    start = time.time()
    print(Style.NORMAL + Fore.CYAN + f"initializing", end="")

    # imports here, because they're slow
    from aitg.model import load_t5_model
    from aitg.util import multiline_in, get_compute_device
    from aitg.gens.t5_generator import T5Generator

    print(f"[{get_compute_device()[1]}]...")

    print(Style.DIM + Fore.RESET + f"[dbg] init in: {time.time() - start:.2f}s")

    start = time.time()
    print(Style.NORMAL + Fore.CYAN + "loading model...")
    ai = load_t5_model(model_dir)
    print(
        Style.DIM
        + Fore.RESET
        + f"[dbg] finished loading in: {time.time() - start:.2f}s"
    )
    print(Style.DIM + Fore.RESET + f"[dbg] model: {ai.model_name} ({ai.model_type})")

    # prompt
    t5gen = T5Generator(ai)
    while True:
        print(Style.NORMAL + Fore.WHITE + "\nprompt" + Fore.GREEN + ":")
        prompt = multiline_in().strip()

        if prompt.strip() == "":
            # done
            break

        print(Style.NORMAL + Fore.WHITE + "□\n――――――――――")
        print(Style.DIM + Fore.RESET + f"generating...", end="")

        start = time.time()

        output = t5gen.generate(
            text=prompt,
            min_length=min_length,
            max_length=max_length,
            temperature=temp,
            typical_p=typical_p,
            max_time=max_time,
        )

        generation_time = time.time() - start
        total_gen_num = len(output.tokens)
        print(
            Style.DIM
            + Fore.RESET
            + f"[{output.num_new}/{total_gen_num}] ({generation_time:.2f}s/{(output.num_new/generation_time):.2f}tps)"
        )
        
        
        print(Style.NORMAL + Fore.MAGENTA + f"{ai.filter_text(output.text)}", end="")
        print(Style.DIM + Fore.RESET + "□")
        if output.num_new == 0:
            # no more tokens
            print(Style.DIM + Fore.RED + "□ □ □")
        print("\n")

def main():
    typer.run(cli)

if __name__ == "__main__":
    main()
