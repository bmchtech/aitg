from aitg_host.util import multiline_in, count_prompt_tokens, str_to_ids, ids_to_toks, str_to_toks, toks_to_str
from math import floor
from aitg_host.raw_generator import raw_generate

class SlidingGenerator:
    def __init__(self, ai, max_length, context_amount):
        self.ai = ai
        self.max_length = max_length
        self.context_amount = context_amount
        self.token_log = []

    def next_context(self):
        # context
        context_len = floor(self.max_length * self.context_amount)
        context_tokens = self.token_log[-context_len:].copy()
        context_prompt = toks_to_str(self.ai, context_tokens)

        return context_prompt, context_tokens
    
    def generate(
        self,
        prompt: str = "",
        fresh: bool = True,
        prepend_bos: bool = None,
        min_length: int = None,
        max_length: int = 256,
        temperature: float = 0.7,
        do_sample: bool = True,
        return_as_list: bool = False,
        seed: int = None,
        pad_token_id: str = None,
        schema: str = False,
        normalize_key: bool = True,
        use_cache: bool = True,
        lstrip: bool = True,
        skip_special_tokens: bool = True,
        **kwargs
    ):
        context_prompt = ''
        context_toks = []
        if fresh:
            self.token_log = []
        else:
            context_prompt, context_toks = self.next_context()

        # full prompt
        full_prompt = context_prompt + prompt

        prompt_tokens = str_to_toks(self.ai, full_prompt)
        # self.token_log.extend(prompt_tokens)

        # gen
        gen_txt, gen_toks = raw_generate(self.ai, 
            max_length=max_length,
            min_length=min_length,
            seed=seed,
            prompt=full_prompt,
            temperature=temperature,
            **kwargs,
        )

        # add the input+output sequence from the model
        # exclude context tokens (because they already are included)
        self.token_log.extend(gen_toks[len(context_toks):])
        # count how many new toks were added (to see if we're at the end)
        num_new_toks = len(gen_toks) - len(prompt_tokens)
        return gen_txt, gen_toks, num_new_toks