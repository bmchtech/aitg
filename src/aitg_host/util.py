import sys

def multiline_in():
    return sys.stdin.read()

def count_prompt_tokens(ai, prompt):
    prompt_tensors = ai.tokenizer(text=prompt, return_tensors="pt")
    num_tokens = list(prompt_tensors["input_ids"].shape)[1]
    return num_tokens

def str_to_ids(ai, text):
    return ai.tokenizer(text=text)['input_ids']

def ids_to_toks(ai, ids, skip_special_tokens=True):
    return ai.tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=skip_special_tokens)

def toks_to_str(ai, toks):
    return ai.tokenizer.convert_tokens_to_string(toks)