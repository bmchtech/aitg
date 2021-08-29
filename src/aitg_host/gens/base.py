import re

class BaseGenerator:
    def __init__(self, ai):
        self.ai = ai

    def count_tokens(self, text):
        return len(self.str_to_ids(text))

    def str_to_ids(self, text):
        return self.ai.tokenizer(text=text).input_ids

    def ids_to_toks(self, ids, skip_special_tokens=True):
        return self.ai.tokenizer.convert_ids_to_tokens(
            ids, skip_special_tokens=skip_special_tokens
        )

    def str_to_toks(self, text):
        ids = self.str_to_ids(text)
        return self.ids_to_toks(ids)

    def toks_to_str(self, toks):
        return self.ai.tokenizer.convert_tokens_to_string(toks)

    def lstrip_texts(self, texts):
        # handle stripping tokenization spaces w/ regex
        return [re.sub(r"^\s+", "", text) for text in texts]