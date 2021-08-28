import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, logging
from aitg_host.textgen.base_generator import BaseGenerator
from types import SimpleNamespace


class SummarizerAI:
    def __init__(self, model_folder: str, to_device, verbose=False):
        self.model_folder = model_folder
        self.device = to_device

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

    def str_to_ids(self, text):
        # custom tokenizer invocation (because of max length)
        return self.ai.tokenizer(text=text, max_length=1024, truncation=True).input_ids

    def generate(
        self,
        prompt: str = "",
        min_length: int = None,
        max_length: int = 256,
        **kwargs
    ):
        # encode
        article = prompt
        article_tensors = self.ai.tokenizer(text=article, return_tensors="pt", max_length=1024, truncation=True)
        input_ids = article_tensors.input_ids.to(self.ai.device)

        # generate
        output_ids = self.ai.model.generate(
            input_ids,
            min_length=min_length,
            max_length=max_length,
            **kwargs,
        )

        # decode
        output_seq = output_ids.squeeze()
        output_text = self.ai.tokenizer.decode(output_seq, skip_special_tokens=True)
        output_tokens = self.ai.tokenizer.convert_ids_to_tokens(output_seq, skip_special_tokens=True)

        return SimpleNamespace(
            text=output_text,
            tokens=output_tokens,
            seq=output_seq,
            num_new=len(output_tokens),
            prompt_ids=input_ids.tolist()[0],
            # probs=probs[0, :, :].tolist(),
        )
