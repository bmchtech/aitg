import time
import os
from aitg_host.util import multiline_in, count_prompt_tokens, str_to_ids, ids_to_toks, str_to_toks, toks_to_str
from math import floor
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
    token_log = []
    while True:
        print(Style.NORMAL + Fore.WHITE + "\nprompt" + Fore.GREEN + ':')
        prompt = multiline_in()
        gen_type = 'generating'

        if prompt == '' and len(token_log) > 0:
            gen_type = 'continuing'
            context_portion = 0.5
            context_len = floor(max_length * context_portion)
            context_tokens = token_log[-context_len:].copy()
            print(f'using context[{context_len}]: {toks_to_str(ai, context_tokens)}')
            prompt = toks_to_str(ai, context_tokens)

        print(Style.NORMAL + Fore.WHITE + "□\n――――――――――")
        print(Style.DIM + Fore.RESET + f"{gen_type}...", end='')
        prompt_tokens = str_to_toks(ai, prompt)
        token_log.extend(prompt_tokens)

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

        token_log.extend(gen_toks)
        print(Style.DIM + Fore.RESET + f"[{len(gen_toks)}] ({time.time() - start:.2f}s)")
        if (gen_type == 'continuing'):
            print(Style.NORMAL + Fore.MAGENTA + f"{toks_to_str(ai, token_log)}")
        else:
            print(Style.NORMAL + Fore.MAGENTA + f"{gen_txt}")

        # print(token_log)

        # # print half of tokens
        # half_toks = gen_toks.copy()
        # half_toks = half_toks[:(len(half_toks) // 2)]
        # print(gen_toks)
        # print(half_toks)
        # print(toks_to_str(ai, half_toks))

        # # print tokenized result
        # res_ids = str_to_ids(ai, gen_txt)
        # print(f"res ids: {res_ids}")
        # res_toks = ids_to_toks(ai, res_ids)
        # print(f"res toks: {res_toks}")
        # print(f"res outstr: {toks_to_str(ai, res_toks)}")

        print('\n')


def main():
    typer.run(cli)


if __name__ == "__main__":
    main()