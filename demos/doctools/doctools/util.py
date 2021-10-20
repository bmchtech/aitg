import itertools
import os
import sys

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


# file contents
def read_file(path):
    with open(path) as f:
        return f.read()

def batch_list(input, size):
    it = iter(input)
    item = list(itertools.islice(it, size))
    while item:
        yield item
        item = list(itertools.islice(it, size))