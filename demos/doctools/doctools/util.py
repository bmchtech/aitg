import itertools
import os
import sys

import lzma

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


# file contents
def read_file(path):
    with open(path) as f:
        return f.read()

def listdir_recursive(path):
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            yield os.path.join(dirpath, filename)

def batch_list(input, size):
    it = iter(input)
    item = list(itertools.islice(it, size))
    while item:
        yield item
        item = list(itertools.islice(it, size))

LZMA_OPTIONS = {
    "format": lzma.FORMAT_XZ,
}

def lzma_compress(data):
    lzc = lzma.LZMACompressor(
        **LZMA_OPTIONS,
        check=lzma.CHECK_CRC32,
        preset=lzma.PRESET_EXTREME,
    )

    out = lzc.compress(data)
    out += lzc.flush()
    return out

def lzma_decompress(data):
    lzc = lzma.LZMADecompressor(**LZMA_OPTIONS)
    return lzc.decompress(data)
