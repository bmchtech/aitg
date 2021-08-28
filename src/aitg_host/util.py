import sys
import torch


def multiline_in():
    return sys.stdin.read()


def get_compute_device():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        return device, "gpu"
    else:
        device = torch.device("cpu")
        return device, "cpu"
