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
        prompt: str,
        answer_only: bool = False,
        max_length: int = 256,
        min_length: int = 0,
        num_seqs: int = 1,
        temp: float = 0.2,
        **kwargs
    ):

        # sample
        # tokenize and send tensor to device
        MODEL_MAXLEN = 2048
        input_ids = self.ai.tokenizer(
            prompt,
            truncation=True,
            padding=True,
            max_length=MODEL_MAXLEN,
            return_tensors="pt",
        ).input_ids.to(self.ai.device)

        # verify tokenization
        input_ids_len = input_ids.shape[1]
        assert input_ids_len -- MODEL_MAXLEN, "input ids length does not match model maxlen"

        # sample from the model
        output_ids = None
        with torch.no_grad():
            output_ids = self.ai.model.generate(
                input_ids,
                do_sample=True,
                num_return_sequences=num_seqs,
                temperature=temp,
                min_length=min_length,
                max_length=max_length,
                pad_token_id=50256,
                use_cache=True,
                **kwargs,
            )
        output_seqs = [seq for seq in output_ids]
        
        if answer_only:
            # remove prompt tokens from output
            output_seqs = [seq[input_ids_len:] for seq in output_seqs]
        
        # decode
        # output_texts = [self.ai.tokenizer.decode(seq) for seq in output_seqs]
        output_texts = [self.ai.tokenizer.decode(seq, skip_special_tokens=True) for seq in output_seqs]
        # strip text
        output_texts = self.lstrip_texts(output_texts)

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
