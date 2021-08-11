import time
import os
from aitg_host.util import multiline_in, count_prompt_tokens, str_to_ids, ids_to_toks, str_to_toks, toks_to_str
from math import floor
import typer
import colorama
from colorama import Fore, Back, Style
from aitg_host.raw_generator import raw_generate
from aitg_host.sliding_generator import SlidingGenerator

MODEL_DIR = os.environ["MODEL"]


def cli(
    temp: float = 0.9,
    max_length: int = 256,
    min_length: int = 0,
    context_amount: float = 0.5,
    seed: int = None,
    top_p: float = 0.9,
    top_k: int = 0,
    repetition_penalty: float = 1.0,
    length_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
    optimize: bool = True,
):
    colorama.init()

    start = time.time()
    print(Style.NORMAL + Fore.CYAN + "initializing...")
    from aitg_host.model import load_model
    print(Style.DIM + Fore.RESET + f"[dbg] init in: {time.time() - start:.2f}s")

    start = time.time()
    print(Style.NORMAL + Fore.CYAN + "loading model...")
    ai = load_model(MODEL_DIR, optimize)
    print(Style.DIM + Fore.RESET + f"[dbg] finished loading in: {time.time() - start:.2f}s")

    # prompt
    slidegen = SlidingGenerator(ai, max_length, context_amount)
    while True:
        print(Style.NORMAL + Fore.WHITE + "\nprompt" + Fore.GREEN + ':')
        prompt = multiline_in()
        gen_type = 'generating'
        is_fresh = True

        if prompt == '' and len(slidegen.token_log) > 0:
            gen_type = 'continuing'
            is_fresh = False
            context, context_toks = slidegen.next_context()
            print(Style.DIM + Fore.RESET + f'context[{len(context_toks)}]: {context}', end='')

        print(Style.NORMAL + Fore.WHITE + "□\n――――――――――")
        print(Style.DIM + Fore.RESET + f"{gen_type}...", end='')

        start = time.time()
        
        gen_txt, gen_toks, num_new = slidegen.generate(
            prompt,
            fresh=is_fresh,
            min_length=min_length,
            max_length=max_length,
            seed=seed,
            temperature=temp,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size
        )

        print(Style.DIM + Fore.RESET + f"[{len(gen_toks)}] ({time.time() - start:.2f}s)")
        # print(Style.NORMAL + Fore.MAGENTA + f"{gen_toks}")
        if (is_fresh):
            print(Style.NORMAL + Fore.MAGENTA + f"{gen_txt}", end='')
        else:
            print(Style.NORMAL + Fore.MAGENTA + f"{toks_to_str(ai, slidegen.token_log)}", end='')
        print(Style.DIM + Fore.RESET + "□")
        if (num_new == 0):
            # no more tokens
            print(Style.DIM + Fore.RED + '□ □ □')
        print('\n')

def main():
    typer.run(cli)


if __name__ == "__main__":
    main()