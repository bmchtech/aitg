from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, logging


class BartSummarizerAI:
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

        self.context_window = 1024
