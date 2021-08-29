import torch
import torch.nn.functional as F
from aitg_host.gens.base import BaseGenerator
from types import SimpleNamespace
from typing import List


class EmbedGenerator(BaseGenerator):
    def __init__(self, ai):
        super().__init__(ai)

    def str_to_ids(self, text):
        # custom tokenizer invocation
        return self.ai.tokenizer(text=text, padding=True, truncation=True).input_ids

    def mean_pooling(self, model_output, attention_mask):
        # take attention mask into account for correct averaging
        token_embeddings = model_output[
            0
        ]  # first element of model_output contains all token embeddings
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def generate(self, texts: List[str], **kwargs):
        # encode
        input_tensors = self.ai.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        )

        # Compute token embeddings
        with torch.no_grad():
            model_output = self.ai.model(**input_tensors)

        # print('output:', model_output)

        # Perform pooling. In this case, max pooling.
        sentence_embeddings = self.mean_pooling(
            model_output, input_tensors["attention_mask"]
        )

        # print("sentence embeddings:".sentence_embeddings)

        # # try showing similiarity
        # similarity = F.cosine_similarity(
        #     sentence_embeddings[0], sentence_embeddings[1], dim=0
        # )
        # print("similarity:", similarity)

        return SimpleNamespace(
            embeddings=sentence_embeddings.tolist(),
        )
