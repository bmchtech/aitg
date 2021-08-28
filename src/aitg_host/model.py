import os.path
from aitg_host.util import get_compute_device
import importlib.util
import json

def import_pymodule(module_name, module_path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    loaded_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded_module)
    return loaded_module

def ensure_model_dir(load_path):
    # this is a LOCAL model path
    if not os.path.isdir(load_path):
        raise OSError(f'model path is not a valid directory: {load_path}')

def load_common_ext(ai, load_path):
    # try to load model name
    with open(os.path.join(load_path, 'config.json')) as cfg_f:
        cfg_data = json.load(cfg_f)
        if 'model_friendly_id' in cfg_data:
            ai.model_name = cfg_data['model_friendly_id']
        else:
            # use the dirname as fallback
            ai.model_name = os.path.basename(load_path)

    ai.filter_text = lambda x: x # default
    # try loading filter
    filter_module_path = os.path.join(load_path, 'filter.py')
    if os.path.isfile(filter_module_path):
        # load module func
        filter_module = import_pymodule('aitg_model.filter', filter_module_path)
        # print(filter_module.filter_text)
        ai.filter_text = filter_module.filter_text

def load_gpt_model(load_path):
    from aitextgen import aitextgen

    ai = None
    use_gpu = get_compute_device()[1] == "gpu"

    ensure_model_dir(load_path)

    # load model
    ai = aitextgen(model_folder=load_path, to_gpu=use_gpu)

    load_common_ext(ai, load_path)

    return ai

def load_bart_summarizer_model(load_path):
    from aitg_host.textgen.summarizer import SummarizerAI

    ai = None
    device, device_type = get_compute_device()

    ensure_model_dir(load_path)

    # load model
    ai = SummarizerAI(model_folder=load_path, to_device=device)

    load_common_ext(ai, load_path)

    return ai
