from transformers import GPT2Tokenizer, AutoModelForCausalLM, logging

class SFCodegenAI:
    def __init__(self, model_folder: str, to_device, verbose=False):
        self.model_folder = model_folder
        self.device = to_device

        if not verbose:
            logging.set_verbosity_warning()

        # load model
        self.tokenizer = GPT2Tokenizer.from_pretrained(
            model_folder, local_files_only=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_folder, local_files_only=True, trust_remote_code=True,
            low_cpu_mem_usage=True,
        ).to(to_device)
