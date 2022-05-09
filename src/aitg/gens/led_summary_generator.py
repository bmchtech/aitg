import torch
from aitg.gens.base import BaseGenerator
from types import SimpleNamespace


class LedSummaryGenerator(BaseGenerator):
    def __init__(self, ai):
        super().__init__(ai)

    def str_to_ids(self, text):
        # custom tokenizer invocation (because of max length)
        return self.ai.tokenizer(text, return_tensors="pt").input_ids

    def generate(
        self,
        article: str,
        min_length: int = None,
        max_length: int = None,
        lrstrip: bool = True,
        **kwargs
    ):
        # encode
        article_tensors = self.ai.tokenizer(article, return_tensors="pt")
        input_ids = article_tensors.input_ids.to(self.ai.device)

        # attention mask
        global_attention_mask = torch.zeros_like(input_ids)
        # set global_attention_mask on first token
        global_attention_mask[:, 0] = 1

        # generate
        model_output = self.ai.model.generate(
            input_ids,
            global_attention_mask=global_attention_mask,
            return_dict_in_generate=True,
            min_length=min_length,
            max_length=max_length,
            early_stopping=True,
        )

        # print('model output:', model_output)

        # decode
        output_seqs = [seq for seq in model_output.sequences]
        output_texts = [
            self.ai.tokenizer.decode(seq, skip_special_tokens=True)
            for seq in output_seqs
        ]
        output_tokens = [
            self.ai.tokenizer.convert_ids_to_tokens(seq, skip_special_tokens=True)
            for seq in output_seqs
        ]

        if lrstrip:
            output_texts = self.lstrip_texts(output_texts)
            output_texts = [text.strip() for text in output_texts]

        return SimpleNamespace(
            text=output_texts[0],
            tokens=output_tokens[0],
            seq=output_seqs[0],
            num_new=len(output_tokens[0]),
            num_prompt_tokens=len(input_ids.tolist()[0]),
            prompt_ids=input_ids.tolist()[0],
        )
