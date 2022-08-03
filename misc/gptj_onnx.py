import time
import sys
import typer

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from optimum.onnxruntime import ORTModelForCausalLM
from optimum.onnxruntime import ORTOptimizer
from optimum.onnxruntime.configuration import OptimizationConfig

def multiline_in(prompt=''):
    print(prompt, end='')
    sys.stdout.flush()
    return sys.stdin.read()

def cli(
    model_path: str,
):
    print(f"loading model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = ORTModelForCausalLM.from_pretrained(model_path)
    print(f"loaded model")

    # 1. set pad token
    tokenizer.pad_token = tokenizer.eos_token

    # 2. add special tokens
    def include_whitespace(n_min=2, n_max=20, as_special_tokens=False):
        tokenizer.add_tokens([' ' * n for n in reversed(range(n_min, n_max))], special_tokens=as_special_tokens)


    def include_tabs(n_min=2, n_max=20, as_special_tokens=False):
        tokenizer.add_tokens(['\t' * n for n in reversed(range(n_min, n_max))], special_tokens=as_special_tokens)
    
    # add tokens for whitespace and tabs
    include_whitespace(n_min=2, n_max=32, as_special_tokens=False)
    include_tabs(n_min=2, n_max=10, as_special_tokens=False)
    
    user_input = multiline_in("prompt: ")
    while user_input.strip() != "":
        # do inference
        print('\ngenerating...')
        gen_start_time = time.time()
        inputs = tokenizer(
            user_input,
            truncation=True,
            padding=True,
            max_length=1024,
            return_tensors='pt',
        )
        input_ids_len = inputs.input_ids.shape[1]
        max_new = 16
        gen_tokens = model.generate(
            **inputs,
            do_sample=True, temperature=0.2, top_p = 0.95,
            max_length = input_ids_len + max_new,
            use_cache=True,
            pad_token_id=50256,
        )
        gen_elapsed_time = time.time() - gen_start_time
        print(tokenizer.batch_decode(gen_tokens)[0])
        print('generation time: {:.2f}s'.format(gen_elapsed_time))
        print()
        user_input = multiline_in("prompt: ")

def main():
    typer.run(cli)

if __name__ == "__main__":
    main()
