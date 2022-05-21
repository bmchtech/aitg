import os.path
from aitg.util import get_compute_device
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
        raise NotADirectoryError(f"model path is not a valid directory: {load_path}")

    # add checks to make sure the model files are there
    if not os.path.exists(load_path + "/pytorch_model.bin") and not os.path.exists(load_path + "/pytorch_model.bin.index.json"):
        raise FileNotFoundError(f"model directory is missing pytorch model")
    if not os.path.exists(load_path + "/config.json"):
        raise FileNotFoundError(f"model directory is missing config file")


def load_common_ext(ai, load_path):
    # try to load model name
    with open(os.path.join(load_path, "config.json")) as cfg_f:
        cfg_data = json.load(cfg_f)

        # get model metadata
        ai.model_name = cfg_data.get("model_friendly_id", os.path.basename(load_path))
        ai.model_type = cfg_data.get("model_type", "unknown")

    ai.filter_text = lambda x: x  # default
    # try loading filter
    filter_module_path = os.path.join(load_path, "filter.py")
    if os.path.isfile(filter_module_path):
        # load module func
        filter_module = import_pymodule("aitg_model.filter", filter_module_path)
        # print(filter_module.filter_text)
        ai.filter_text = filter_module.filter_text


def load_gpt_model(load_path):
    from aitextgen import aitextgen

    use_gpu = get_compute_device()[1] == "gpu"

    ensure_model_dir(load_path)

    # load model
    ai = aitextgen(model_folder=load_path, to_gpu=use_gpu)

    load_common_ext(ai, load_path)

    return ai


def load_bart_summarizer_model(load_path):
    from aitg.models.bart_summarizer import BartSummarizerAI

    ensure_model_dir(load_path)

    # load model
    ai = BartSummarizerAI(model_folder=load_path, to_device=get_compute_device()[0])

    load_common_ext(ai, load_path)

    return ai


def load_led_summarizer_model(load_path):
    from aitg.models.led__summarizer import LedSummarizerAI

    ensure_model_dir(load_path)

    # load model
    ai = LedSummarizerAI(model_folder=load_path, to_device=get_compute_device()[0])

    load_common_ext(ai, load_path)

    return ai


def load_bart_classifier_model(load_path):
    from aitg.models.bart_classifier import BartClassifierAI

    ensure_model_dir(load_path)

    # load model
    ai = BartClassifierAI(model_folder=load_path, to_device=get_compute_device()[0])

    load_common_ext(ai, load_path)

    return ai


def load_sentence_embed_model(load_path):
    from aitg.models.sentence_embed import SentenceEmbedAI

    ensure_model_dir(load_path)

    # load model
    ai = SentenceEmbedAI(model_folder=load_path, to_device=get_compute_device()[0])

    load_common_ext(ai, load_path)

    return ai


def load_question_answer_model(load_path):
    from aitg.models.question_answer import QuestionAnswerAI

    ensure_model_dir(load_path)

    # load model
    ai = QuestionAnswerAI(model_folder=load_path, to_device=get_compute_device()[0])

    load_common_ext(ai, load_path)

    return ai

def load_t5_model(load_path):
    from aitg.models.t5 import T5AI

    ensure_model_dir(load_path)

    # load model
    ai = T5AI(model_folder=load_path, to_device=get_compute_device()[0])

    load_common_ext(ai, load_path)

    return ai

def load_sfcodegen_model(load_path):
    from aitg.models.sfcodegen import SFCodegenAI

    ensure_model_dir(load_path)

    # load model
    ai = SFCodegenAI(model_folder=load_path, to_device=get_compute_device()[0])

    load_common_ext(ai, load_path)

    return ai


import typer


def download_model(
    model_arch: str = typer.Argument(
        ...,
        help="the model architecture (for example, AutoModel or T5ForConditionalGeneration)",
    ),
    model_id: str = typer.Argument(
        ...,
        help="the huggingface model id. usually looks like @organization/some-model",
    ),
    tokenizer_arch: str = 'AutoTokenizer',
    path: str = typer.Argument(..., help="the output path to save the model to"),
    trust_remote: bool = False,
    low_mem: bool = False,
):
    """
    download a model id from huggingface, and save to a local path
    """
    # ensure this is huggingface
    if model_id.startswith("@"):
        # this is a HUGGINGFACE model path (download from repo)
        model_id = model_id[1:]

        _transformers_mod = __import__("transformers")
        _model_class = getattr(_transformers_mod, model_arch)
        _tokenizer_class = getattr(_transformers_mod, tokenizer_arch)

        print("using model architecture:", _model_class)
        print("using tokenizer architecture:", _tokenizer_class)

        # grab both
        print(f"getting model: {model_id}")
        if trust_remote:
            model = _model_class.from_pretrained(model_id, trust_remote_code=True)
        elif low_mem:
            model = _model_class.from_pretrained(model_id, low_cpu_mem_usage=True)
        else:
            model = _model_class.from_pretrained(model_id)
        print(f"getting tokenizer: {model_id}")
        tokenizer = _tokenizer_class.from_pretrained(model_id)

        # save both
        print(f"saving to: {path}")
        model.save_pretrained(save_directory=path)
        tokenizer.save_pretrained(save_directory=path)
    else:
        print("preface huggingface model id with @")


def download_model_main():
    typer.run(download_model)

if __name__ == "__main__":
    download_model_main()