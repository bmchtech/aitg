import time
import os
from aitg_host.util import multiline_in, get_compute_device
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
    reuse_session: bool = False,
):
    colorama.init()

    start = time.time()
    print(Style.NORMAL + Fore.CYAN + f"initializing[{get_compute_device()}]...")
    from aitg_host.model import load_model
    print(Style.DIM + Fore.RESET + f"[dbg] init in: {time.time() - start:.2f}s")

    start = time.time()
    print(Style.NORMAL + Fore.CYAN + "loading model...")
    ai = load_model(MODEL_DIR, optimize)
    print(Style.DIM + Fore.RESET + f"[dbg] finished loading in: {time.time() - start:.2f}s")
    print(Style.DIM + Fore.RESET + f"[dbg] model: {ai.model_name}")

    # prompt
    slidegen = SlidingGenerator(ai)
    while True:
        print(Style.NORMAL + Fore.WHITE + "\nprompt" + Fore.GREEN + ':')
        prompt = multiline_in()
        gen_type = 'generating'
        is_fresh = True
        
        # if reuse enabled, go not fresh by default
        if reuse_session:
            is_fresh = False

        if prompt == '' and len(slidegen.token_log) > 0:
            gen_type = 'continuing'
            is_fresh = False
            context, context_toks = slidegen.next_context(max_length, context_amount)
            print(Style.DIM + Fore.RESET + f'context[{len(context_toks)}]: {context}', end='')

        print(Style.NORMAL + Fore.WHITE + "□\n――――――――――")
        print(Style.DIM + Fore.RESET + f"{gen_type}...", end='')

        start = time.time()
        
        gen_txt, gen_toks, num_new = slidegen.generate(
            prompt=prompt,
            fresh=is_fresh,
            context_amount=context_amount,
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

        # gen_txt, gen_toks = slidegen.generate_rounds(
        #     prompt,
        #     min_length=min_length,
        #     max_length=max_length,
        #     temperature=temp,
        # )

        generation_time = time.time() - start
        total_gen_num = len(gen_toks)
        print(Style.DIM + Fore.RESET + f"[{num_new}/{total_gen_num}] ({generation_time:.2f}s/{(num_new/generation_time):.2f}tps)")
        # print(Style.NORMAL + Fore.MAGENTA + f"{gen_toks}")
        if (is_fresh):
            print(Style.NORMAL + Fore.MAGENTA + f"{ai.filter_text(gen_txt)}", end='')
        else:
            print(Style.NORMAL + Fore.MAGENTA + f"{ai.filter_text(slidegen.toks_to_str(slidegen.token_log))}", end='')
        print(Style.DIM + Fore.RESET + "□")
        if (num_new == 0):
            # no more tokens
            print(Style.DIM + Fore.RED + '□ □ □')
        print('\n')

def main():
    typer.run(cli)


if __name__ == "__main__":
    main()
