from transformers import AutoTokenizer, AutoModelForCausalLM
from optimum.onnxruntime import ORTModelForCausalLM
from optimum.onnxruntime import ORTOptimizer
from optimum.onnxruntime.configuration import OptimizationConfig
import torch
import typer

def cli(
    model_path: str,
):
    print(f"loading model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = ORTModelForCausalLM.from_pretrained(model_path)
    print(f"loaded model")

    # inputs = tokenizer("My name is Philipp and I live in Germany.", return_tensors="pt")
    # gen_tokens = model.generate(
    #     **inputs, do_sample=True, temperature=0.9, min_length=20, max_length=20
    # )
    # tokenizer.batch_decode(gen_tokens)
    
    user_input = input("prompt: ")
    while user_input.strip() != "":
        # do inference
        inputs = tokenizer(user_input, return_tensors="pt")
        gen_tokens = model.generate(
            **inputs, do_sample=True, temperature=0.9, max_length=32
        )
        print(tokenizer.batch_decode(gen_tokens))
        user_input = input("prompt: ")

def main():
    typer.run(cli)

if __name__ == "__main__":
    main()
