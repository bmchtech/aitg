import torch
from aitg.gens.base import BaseGenerator
from types import SimpleNamespace
import re

class SFCodegenGenerator(BaseGenerator):
    def __init__(self, ai):
        super().__init__(ai)

    def str_to_ids(self, text):
        return self.ai.tokenizer(text=text).input_ids

    # def truncate(self, completion):
    #     def find_re(string, pattern, start_pos):
    #         m = pattern.search(string, start_pos)
    #         return m.start() if m else -1

    #     terminals = [
    #         re.compile(r, re.MULTILINE)
    #         for r in
    #         [
    #             '^#',
    #             re.escape('<|endoftext|>'),
    #             "^'''",
    #             '^"""',
    #             '\n\n\n'
    #         ]
    #     ]

    #     prints = list(re.finditer('^print', completion, re.MULTILINE))
    #     if len(prints) > 1:
    #         completion = completion[:prints[1].start()]

    #     defs = list(re.finditer('^def', completion, re.MULTILINE))
    #     if len(defs) > 1:
    #         completion = completion[:defs[1].start()]

    #     start_pos = 0

    #     terminals_pos = [pos for pos in [find_re(completion, terminal, start_pos) for terminal in terminals] if pos != -1]
    #     if len(terminals_pos) > 0:
    #         return completion[:min(terminals_pos)]
    #     else:
    #         return completion

    def generate(
        self,
        context: str,
        max_length: int = 2048,
        max_length_sample: int = 128,
        num_seqs: int = 1,
        temp: float = 0.2,
        top_p: float = 0.95,
        **kwargs
    ):
        
        # sample
        # completion = self.sample(context, num_return_sequences=num_seqs, temp=temp, top_p=top_p, max_length=max_length, max_length_sample=max_length_sample)[0]
        # tokenize and send tensor to device
        input_ids = self.ai.tokenizer(
            context,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors='pt',
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
                max_length=input_ids_len + max_length_sample,
                top_p=top_p,
                pad_token_id=50256,
                use_cache=True,
                **kwargs,
            )
        output_seqs = output_seqs = [seq for seq in output_ids]
        output_texts = [
            self.ai.tokenizer.decode(seq, skip_special_tokens=True)
            for seq in output_seqs
        ]
        output_tokens = [
            self.ai.tokenizer.convert_ids_to_tokens(seq, skip_special_tokens=True)
            for seq in output_seqs
        ]
        # output_texts = self.ai.tokenizer.batch_decode(output_ids[:, input_ids_len:, ...])
        # print('output_texts:', output_texts)
        # print('output_tokens:', output_tokens)
        # truncation = self.truncate(output_texts[0])
        
        # return

        return SimpleNamespace(
            # text=context + truncation,
            text=output_texts[0],
            tokens=output_tokens[0],
            seq=output_seqs[0],
            num_new=len(output_tokens[0]),
            num_prompt_tokens=len(input_ids.tolist()[0]),
            prompt_ids=input_ids.tolist()[0],
            # probs=probs[0, :, :].tolist(),
        )
