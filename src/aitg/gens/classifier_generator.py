import torch
from aitg.gens.base import BaseGenerator
from typing import List
from types import SimpleNamespace
import numpy as np


class ClassifierGenerator(BaseGenerator):
    def __init__(self, ai):
        super().__init__(ai)

        if self.entailment_id == -1:
            print(
                "WARNING: Failed to determine 'entailment' label id from the label2id mapping in the model config. Setting to "
                "-1. Define a descriptive label2id mapping in the model config to ensure correct outputs."
            )

    def str_to_ids(self, text):
        # custom tokenizer invocation (because of max length)
        return self.ai.tokenizer(
            text=text, max_length=self.ai.context_window, truncation=True
        ).input_ids

    @property
    def entailment_id(self):
        for label, ind in self.ai.model.config.label2id.items():
            if label.lower().startswith("entail"):
                return ind
        return -1

    def generate(
        self,
        text: str,
        candidate_labels: List[str],
        hypothesis_template="This example is {}.",
        multi_label=False,
        **kwargs,
    ):
        # encode
        premise = text
        # create text pairs
        text_pairs = []
        for label in candidate_labels:
            hypothesis = hypothesis_template.format(label)
            text_pairs.append([text, hypothesis])

        outputs = []
        # run the model
        for text_pair in text_pairs:
            # create input tensor
            input_tensor = self.ai.tokenizer(
                text_pair[0],
                text_pair[1],
                return_tensors="pt",
                truncation_strategy="only_first",
            )
            input_ids = input_tensor.input_ids.to(self.ai.device)

            # generate
            model_output = self.ai.model(
                input_ids,
                **kwargs,
            )

            outputs.append(model_output.logits.detach().numpy())

        # reshape to [num_seq, num_labels, logits]
        reshaped_outputs = np.array(outputs).reshape(1, len(candidate_labels), -1)
        # print("classification out:", reshaped_outputs.shape, reshaped_outputs)

        if len(candidate_labels) == 1:
            multi_label = True

        if not multi_label:
            # softmax the "entailment" logits over all candidate labels
            entail_logits = reshaped_outputs[..., self.entailment_id]
            scores = np.exp(entail_logits) / np.exp(entail_logits).sum(
                -1, keepdims=True
            )
        else:
            # softmax over the entailment vs. contradiction dim for each label independently
            entailment_id = self.entailment_id
            contradiction_id = -1 if entailment_id == 0 else 0
            entail_contr_logits = reshaped_outputs[
                ..., [contradiction_id, entailment_id]
            ]
            scores = np.exp(entail_contr_logits) / np.exp(entail_contr_logits).sum(
                -1, keepdims=True
            )
            scores = scores[..., 1]

        # use scores[0], aka scores from first seq
        top_inds = list(reversed(scores[0].argsort()))
        top_scores = scores[0][top_inds].tolist()
        top_labels = [candidate_labels[i] for i in top_inds]

        # for i in range(len(top_labels)):
        #     print(f"{top_labels[i]}: {top_scores[i]}") # print score

        return SimpleNamespace(labels=top_labels, scores=top_scores)
