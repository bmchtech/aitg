from math import floor
from aitg.models import gpt
from aitg.gens.base import BaseGenerator

class SlidingGenerator(BaseGenerator):
    def __init__(self, ai):
        super().__init__(ai)
        self.token_log = []

    def next_context(self, max_length, context_amount):
        # context
        context_len = floor(max_length * context_amount)
        context_tokens = self.token_log[-context_len:].copy()
        context_prompt = self.toks_to_str(context_tokens)

        return context_prompt, context_tokens

    def generate(
        self,
        prompt: str = "",
        fresh: bool = True,
        context_amount: float = 0.5,
        prepend_bos: bool = None,
        min_length: int = None,
        max_length: int = 256,
        temperature: float = 0.7,
        seed: int = None,
        lstrip: bool = True,
        skip_special_tokens: bool = True,
        **kwargs
    ):
        # determine context section of prompt
        context_prompt = ""
        context_toks = []
        if fresh:
            self.token_log = []
        else:
            context_prompt, context_toks = self.next_context(max_length, context_amount)

        # full prompt
        full_prompt = context_prompt + prompt

        prompt_tokens = self.str_to_toks(full_prompt)

        # gen
        output = gpt.generate(
            self.ai,
            max_length=max_length,
            min_length=min_length,
            seed=seed,
            prompt=full_prompt,
            temperature=temperature,
            prepend_bos=prepend_bos,
            lstrip=lstrip,
            skip_special_tokens=skip_special_tokens,
            **kwargs,
        )

        # count how many new toks were added
        output.num_new = len(output.tokens) - len(output.prompt_ids)

        # add the input + output sequence from the model
        self.token_log.extend(output.tokens[len(context_toks) :])

        return output

    def generate_rounds(
        self,
        max_rounds: int = 0,
        prompt: str = "",
        min_length: int = None,
        max_length: int = 256,
        temperature: float = 0.7,
        **kwargs
    ):
        # first reset log
        self.token_log = []

        # now do a bunch of rounds
        rounds = 0
        while True:
            text, tokens, num_new = self.generate(
                prompt=prompt,
                fresh=False,
                min_length=min_length,
                max_length=max_length,
                temperature=temperature,
                **kwargs,
            )
            prompt = ""  # clear prompt
            if num_new == 0:
                break
            # print(f'round {rounds}: {text}')
            rounds += 1
            if rounds >= max_rounds:
                break

        # no more new, extract the log
        all_round_tokens = self.token_log.copy()
        all_round_output = self.toks_to_str(all_round_tokens)

        return all_round_output, all_round_tokens
