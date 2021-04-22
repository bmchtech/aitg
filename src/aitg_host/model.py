from aitextgen import aitextgen
from os.path import isdir

def load_model(path, optimize):
    if not isdir(path):
        raise OSError(f'model path is not a valid directory: {path}')

    # load model
    ai = aitextgen(model_folder=path)

    if optimize:
        # optimize
        ai.quantize()

    return ai