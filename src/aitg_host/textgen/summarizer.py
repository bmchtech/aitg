import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, logging
from aitg_host.textgen.base_generator import BaseGenerator


class SummarizerAI:
    def __init__(self, model_folder: str, to_device, verbose = False):
        self.model_folder = model_folder

        if not verbose:
            logging.set_verbosity_warning()

        # load model
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_folder, local_files_only=True
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_folder, local_files_only=True
        ).to(to_device)


class SummaryGenerator(BaseGenerator):
    def __init__(self, ai):
        super().__init__(ai)
