import time
import os
from aitg_host.util import multiline_in, count_prompt_tokens
import typer
import colorama
from colorama import Fore, Back, Style
from aitg_host.raw_generator import raw_generate

MODEL_DIR = os.environ["MODEL"]


def cli(
    temp: float = 0.9,
    max_length: int = 256,
    min_length: int = 0,
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
    while True:
        print(Style.NORMAL + Fore.WHITE + "\nprompt" + Fore.GREEN + ':')
        prompt = multiline_in()
        print(Style.NORMAL + Fore.WHITE + "□\n――――――――――")
        print(Style.DIM + Fore.RESET + "generating...", end='')
        num_tokens = count_prompt_tokens(ai, prompt)

        start = time.time()
        # gen_txt = ai.generate_one(
        #     max_length=max_length,
        #     min_length=min_length,
        #     seed=seed,
        #     prompt=prompt,
        #     temperature=temp,
        #     top_p=top_p,
        #     top_k=top_k,
        #     repetition_penalty=repetition_penalty,
        #     length_penalty=length_penalty,
        #     no_repeat_ngram_size=no_repeat_ngram_size,
        # )
        gen_txt, gen_toks = raw_generate(ai, 
            max_length=max_length,
            min_length=min_length,
            seed=seed,
            prompt=prompt,
            temperature=temp,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )

        print(Style.DIM + Fore.RESET + f"[{len(gen_toks[0])}] ({time.time() - start:.2f}s)")
        print(Style.NORMAL + Fore.MAGENTA + f"{gen_txt}")
        print('\n')


def main():
    typer.run(cli)


if __name__ == "__main__":
    main()