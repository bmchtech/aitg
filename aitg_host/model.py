from aitextgen import aitextgen

def load_model(path, optimize):
    # load model
    ai = aitextgen(model=path+'/pytorch_model.bin', config=path+'/config.json')

    if optimize:
        # optimize
        ai.quantize()

    return ai