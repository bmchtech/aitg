import sys

def multiline_in():
    return sys.stdin.read()

def count_prompt_tokens(ai, prompt):
    prompt_tensors = ai.tokenizer(text=prompt, return_tensors="pt")
    num_tokens = list(prompt_tensors["input_ids"].shape)[1]
    return num_tokens