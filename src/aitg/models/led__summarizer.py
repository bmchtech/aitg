from transformers import LEDForConditionalGeneration, LEDTokenizer, logging


class LedSummarizerAI:
    def __init__(self, model_folder: str, to_device, verbose=False):
        self.model_folder = model_folder
        self.device = to_device

        if not verbose:
            logging.set_verbosity_warning()

        # load model
        self.tokenizer = LEDTokenizer.from_pretrained(
            model_folder, local_files_only=True
        )
        self.model = LEDForConditionalGeneration.from_pretrained(
            model_folder, local_files_only=True
        ).to(to_device)

        # model config
        self.model.config.no_repeat_ngram_size = 3
