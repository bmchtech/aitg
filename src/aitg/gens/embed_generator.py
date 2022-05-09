import torch
import torch.nn.functional as F
from aitg.gens.base import BaseGenerator
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

        input_ids = input_tensors.input_ids.to(self.ai.device)
        attention_mask = input_tensors.attention_mask.to(self.ai.device)

        # Compute token embeddings
        with torch.no_grad():
            model_output = self.ai.model(input_ids)

        # print('output:', model_output)

        # Perform pooling. In this case, max pooling.
        sentence_embeddings = self.mean_pooling(
            model_output, attention_mask
        )

        # print("sentence embeddings:", sentence_embeddings.shape(), sentence_embeddings)

        # # try showing similiarity
        # similarity = F.cosine_similarity(
        #     sentence_embeddings[0], sentence_embeddings[1], dim=0
        # )
        # print("similarity of first two:", similarity)

        # compute similarity matrix
        similarity_mat = F.cosine_similarity(
            sentence_embeddings[None, :, :],
            sentence_embeddings[:, None, :],
            dim=-1,
        )
        # print("similarity mat:", similarity_mat.shape, similarity_mat)
        # print('first two similarity:', similarity_mat[0][1])

        return SimpleNamespace(
            similarity=similarity_mat.tolist(),
            embeddings=sentence_embeddings.tolist(),
        )
