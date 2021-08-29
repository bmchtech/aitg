import os
import sys
import torch


def multiline_in():
    return sys.stdin.read()


def get_compute_device():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        total_mem = torch.cuda.get_device_properties(device).total_memory / 1024**3
        return device, "gpu", total_mem
    else:
        device = torch.device("cpu")
        total_mem = 0
        if sys.platform == "linux":
            total_mem = (os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')) / 1024**3
        return device, "cpu", total_mem
