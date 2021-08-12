from aitextgen import aitextgen
from os.path import isdir
from aitg_host.util import get_compute_device

def load_model(path, optimize):
    ai = None
    use_gpu = get_compute_device() == "gpu"
    # check if path is remote
    if path.startswith('@'):
        # this is a HUGGINGFACE model path (download from repo)
        path = path[1:]

        # load model
        ai = aitextgen(model=path, to_gpu=use_gpu)
    else:
        # this is a LOCAL model path
        if not isdir(path):
            raise OSError(f'model path is not a valid directory: {path}')

        # load model
        ai = aitextgen(model_folder=path, to_gpu=use_gpu)

    # if optimize:
    #     # optimize
    #     ai.quantize()

    return ai