import torch
from aitg_host.gens.base import BaseGenerator
from typing import List
from types import SimpleNamespace


class ClassifierGenerator(BaseGenerator):
    def __init__(self, ai):
        super().__init__(ai)

    def str_to_ids(self, text):
        # custom tokenizer invocation (because of max length)
        return self.ai.tokenizer(
            text=text, max_length=self.ai.context_window, truncation=True
        ).input_ids

    def generate(
        self,
        text: str,
        classes: List[str],
        **kwargs,
    ):
        # encode
        premise = text
        hypothesis = f"This example is {classes[0]}"
        input_tensor = self.ai.tokenizer(
            premise, hypothesis, return_tensors="pt", truncation_strategy="only_first"
        )
        input_ids = input_tensor.input_ids.to(self.ai.device)

        # generate
        model_output = self.ai.model(
            input_ids,
            **kwargs,
        )

        logits = model_output.logits
        print("classification out:", dir(model_output), logits.size(), logits)

        # we throw away "neutral" (dim 1) and take the probability of
        # "entailment" (2) as the probability of the label being true
        entail_contradiction_logits = logits[:, [0, 2]]
        probs = entail_contradiction_logits.softmax(dim=1)
        prob_label_is_true = probs[:, 1]
        label_prob = prob_label_is_true.tolist()[0]

        print("probability:", label_prob)

        return SimpleNamespace(
            # results=label_values
            results=label_prob
        )
