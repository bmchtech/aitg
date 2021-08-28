import sys
import torch

def multiline_in():
    return sys.stdin.read()

def count_tokens(ai, text):
    return len(str_to_ids(ai, text))

def str_to_ids(ai, text):
    return ai.tokenizer(text=text).input_ids

def ids_to_toks(ai, ids, skip_special_tokens=True):
    return ai.tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=skip_special_tokens)

def str_to_toks(ai, text):
    return ids_to_toks(ai, str_to_ids(ai, text))

def toks_to_str(ai, toks):
    return ai.tokenizer.convert_tokens_to_string(toks)

def get_compute_device():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        return device, "gpu"
    else:
        device = torch.device("cpu")
        return device, "cpu"