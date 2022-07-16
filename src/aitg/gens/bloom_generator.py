import torch
from aitg.gens.base import BaseGenerator
from types import SimpleNamespace
import re


class BloomGenerator(BaseGenerator):
    def __init__(self, ai):
        super().__init__(ai)

    def str_to_ids(self, text):
        return self.ai.tokenizer(text=text).input_ids

    def generate(
        self,
        context: str,
        max_length: int = 2048,
        sample_length: int = 128,
        num_seqs: int = 1,
        temp: float = 0.2,
        top_p: float = 0.95,
        **kwargs
    ):

        # sample
        # completion = self.sample(context, num_return_sequences=num_seqs, temp=temp, top_p=top_p, max_length=max_length, sample_length=sample_length)[0]
        # tokenize and send tensor to device
        input_ids = self.ai.tokenizer(
            context,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        ).input_ids.to(self.ai.device)

        # verify tokenization
        input_ids_len = input_ids.shape[1]
        assert input_ids_len < max_length

        # sample from the model
        output_ids = None
        with torch.no_grad():
            output_ids = self.ai.model.generate(
                input_ids,
                do_sample=True,
                num_return_sequences=num_seqs,
                temperature=temp,
                max_length=input_ids_len + sample_length,
                top_p=top_p,
                pad_token_id=50256,
                use_cache=True,
                **kwargs,
            )
        output_seqs = [seq for seq in output_ids]
        output_texts = [self.ai.tokenizer.decode(seq) for seq in output_seqs]

        # decode seqs and prompts
        output_tokens = [
            self.ai.tokenizer.convert_ids_to_tokens(seq) for seq in output_seqs
        ]
        total_token_num = sum([len(seq) for seq in output_tokens])

        prompt_tokens = self.ai.tokenizer.convert_ids_to_tokens(input_ids.tolist()[0])

        # count new tokens
        new_tokens = [seq[input_ids_len:] for seq in output_seqs]

        total_new_token_num = sum([len(seq) for seq in new_tokens])

        # return

        return SimpleNamespace(
            texts=output_texts,
            tokens=output_tokens,
            total_gen_tokens=total_token_num,
            num_new=total_new_token_num,
            num_prompt_tokens=len(input_ids.tolist()[0]),
            prompt_ids=input_ids.tolist()[0],
            prompt_tokens=prompt_tokens,
            # probs=probs[0, :, :].tolist(),
        )
