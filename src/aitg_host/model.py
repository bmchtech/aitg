from aitextgen import aitextgen
from os.path import isdir

def load_model(path, optimize):
    ai = None
    # check if path is remote
    if path.startswith('@'):
        # this is a HUGGINGFACE model path (download from repo)
        path = path[1:]

        # load model
        ai = aitextgen(model=path)
    else:
        # this is a LOCAL model path
        if not isdir(path):
            raise OSError(f'model path is not a valid directory: {path}')

        # load model
        ai = aitextgen(model_folder=path)

    # if optimize:
    #     # optimize
    #     ai.quantize()

    return ai