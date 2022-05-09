import torch
from aitg.gens.base import BaseGenerator
from types import SimpleNamespace

class T5Generator(BaseGenerator):
    def __init__(self, ai):
        super().__init__(ai)

    def str_to_ids(self, text):
        return self.ai.tokenizer(text=text).input_ids

    def generate(
        self,
        text: str,
        min_length: int = None,
        max_length: int = 256,
        lstrip: bool = True,
        **kwargs
    ):
        # encode
        prompt_tensors = self.ai.tokenizer(
            text=text, return_tensors="pt",
        )
        input_ids = prompt_tensors.input_ids.to(self.ai.device)

        # generate
        output_ids = self.ai.model.generate(
            input_ids,
            min_length=min_length,
            max_length=max_length,
            **kwargs,
        )

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
            num_prompt_tokens=len(input_ids.tolist()[0]),
            prompt_ids=input_ids.tolist()[0],
            # probs=probs[0, :, :].tolist(),
        )
