import time
import os
from aitg_host.util import multiline_in
import typer

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
    start = time.time()
    print("initializing...")
    from aitg_host.model import load_model

    print(f"[dbg] init in: {time.time() - start:.2f}s")
    start = time.time()
    print("loading model...")
    ai = load_model(MODEL_DIR, optimize)
    print(f"[dbg] finished loading in: {time.time() - start:.2f}s")

    # prompt
    while True:
        print("\n\n")
        print("prompt: (multi-line, send EOF to end)")
        prompt = multiline_in()
        print("\ngenerating...\n===\n")

        start = time.time()
        gen_txt = ai.generate_one(
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
        print(gen_txt)
        print(f"[dbg] generated in: {time.time() - start:.2f}s")


def main():
    typer.run(cli)


if __name__ == "__main__":
    main()