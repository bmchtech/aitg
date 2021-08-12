import sys
import torch

def multiline_in():
    return sys.stdin.read()

def count_tokens(ai, text):
    return len(str_to_ids(ai, text))

def str_to_ids(ai, text):
    return ai.tokenizer(text=text)['input_ids']

def ids_to_toks(ai, ids, skip_special_tokens=True):
    return ai.tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=skip_special_tokens)

def str_to_toks(ai, text):
    return ids_to_toks(ai, str_to_ids(ai, text))

def toks_to_str(ai, toks):
    return ai.tokenizer.convert_tokens_to_string(toks)

def compute_device():
    device = "gpu" if torch.cuda.is_available() else "cpu"
    return device