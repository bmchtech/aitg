import sys
import os
import subprocess


def build_delatex(cache_dir="/tmp/opendetex"):
    bin_path = cache_dir + "/delatex"
    if os.path.exists(cache_dir) and os.path.isfile(bin_path):
        # already made, skip
        print(f"delatex is already built at {bin_path}")
        return bin_path

    print("building delatex from source")
    os.makedirs(cache_dir, exist_ok=True)
    subprocess.run(
        ["git", "clone", "https://github.com/pkubowicz/opendetex.git", cache_dir]
    )
    subprocess.run(["make"], cwd=cache_dir)
    # try running
    subprocess.run([bin_path, "-v"], cwd=cache_dir)

    print(f"delatex is built at {bin_path}")

    return bin_path


def run_delatex(delatex_bin, text):
    delatex_proc = subprocess.Popen(
        [delatex_bin],
        stdout=subprocess.PIPE,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # write to proc
    delatex_proc.communicate(input=text.encode())

    # read from proc
    out_text, _ = delatex_proc.communicate()

    return out_text.decode()


def clean_scientific_text(text):
    import re

    # 1. remove newlines, and duplicated spaces
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
    text = text.replace("\t", " ")
    text = text.replace("\f", " ")
    text = re.sub(r"\s\s+", " ", text)
    text = text.strip()

    # 2. try to remove LaTeX using delatex
    # note: process_latex is crappy, don't use it
    # text = process_latex(text, destroy_latex=True)
    delatex_bin = build_delatex()
    text = run_delatex(delatex_bin, text)

    # 3. remove citations, they are clutter
    # citations can either look like [1] [2] or [1, 2,...]
    text = re.sub(r"\[\s?([0-9]{0,4}\s?,?\s?)*\s?\]", " ", text)
    # aggressively remove stuff in parenthesis ()
    text = re.sub(r"\([\w\s\.,;]*\)", " ", text)
    # aggressively remove stuff in brackets []
    text = re.sub(r"\[[\w\s\.,;]*\]", " ", text)
    # aggressively remove name et al 2xxx
    text = re.sub(r"\w+\s?(et al)\s?[\.,]?\s?[0-9]{0,4}", " ", text)

    # a final trim
    text = re.sub(r"\s\s+", " ", text)
    text = text.strip()

    return text


def main():
    text = sys.stdin.read()

    # delatex_bin = build_delatex()
    # output_text = run_delatex(delatex_bin, text)

    output_text = clean_scientific_text(text)
    print("out:", output_text)


if __name__ == "__main__":
    main()
