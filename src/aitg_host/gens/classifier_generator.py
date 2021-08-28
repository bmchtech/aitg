import torch
from aitg_host.gens.base import BaseGenerator
from typing import List
from types import SimpleNamespace

class ClassifierGenerator(BaseGenerator):
    def __init__(self, ai):
        super().__init__(ai)

    def str_to_ids(self, text):
        # custom tokenizer invocation (because of max length)
        return self.ai.tokenizer(text=text, max_length=self.ai.context_window, truncation=True).input_ids

    def generate(
        self,
        text: str,
        classes: List[str],
        min_length: int = None,
        max_length: int = 256,
        lstrip: bool = True,
        **kwargs
    ):
        # encode
        premise = text
        hypothesis = f'This example is {classes[0]}'
        input_tensor = self.ai.tokenizer(
            premise, hypothesis, return_tensors="pt", truncation_strategy='only_first'
        )
        input_ids = input_tensor.input_ids.to(self.ai.device)

        # generate
        model_output = self.ai.model(
            input_ids,
            **kwargs,
        )

        print('classification out:', model_output)

        # decode
        output_seqs = [seq for seq in output_ids]
        output_texts = [
            self.ai.tokenizer.decode(seq, skip_special_tokens=True)
            for seq in output_seqs
        ]
        output_tokens = [
            self.ai.tokenizer.convert_ids_to_tokens(seq, skip_special_tokens=True)
            for seq in output_seqs
        ]

        if lstrip:
            output_texts = self.lstrip_texts(output_texts)

        return SimpleNamespace(
            text=output_texts[0],
            tokens=output_tokens[0],
            seq=output_seqs[0],
            num_new=len(output_tokens[0]),
            prompt_ids=input_ids.tolist()[0],
            # probs=probs[0, :, :].tolist(),
        )
