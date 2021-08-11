from aitg_host.util import count_tokens, str_to_ids, ids_to_toks, str_to_toks, toks_to_str

class BaseGenerator:
    def __init__(self, ai):
        self.ai = ai

    def count_tokens(self, text):
        return count_tokens(self.ai, text)

    def str_to_ids(self, text):
        return count_tokens(self.ai, text)

    def ids_to_toks(self, ids, skip_special_tokens=True):
        return count_tokens(self.ai, ids, skip_special_tokens=skip_special_tokens)

    def str_to_toks(self, text):
        return count_tokens(self.ai, text)

    def toks_to_str(self, toks):
        return count_tokens(self.ai, toks)