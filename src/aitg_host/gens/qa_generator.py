import torch
import torch.nn.functional as F
from aitg_host.gens.base import BaseGenerator
from types import SimpleNamespace
from typing import List


class QuestionAnswerGenerator(BaseGenerator):
    def __init__(self, ai):
        super().__init__(ai)

    def str_to_ids(self, text):
        # custom tokenizer invocation
        return self.ai.tokenizer(text=text, add_special_tokens=True).input_ids

    def generate(self, text: str, questions: List[str], **kwargs):
        gen_answers = []
        for question in questions:
            # encode
            input_tensor = self.ai.tokenizer(
                question, text, add_special_tokens=True, return_tensors="pt"
            )
            input_ids = input_tensor.input_ids.to(self.ai.device)

            outputs = self.ai.model(input_ids, **kwargs)
            answer_start_scores = outputs.start_logits
            answer_end_scores = outputs.end_logits

            answer_start = torch.argmax(
                answer_start_scores
            )  # Get the most likely beginning of answer with the argmax of the score
            answer_end = (
                torch.argmax(answer_end_scores) + 1
            )  # Get the most likely end of answer with the argmax of the score

            # print('answer locs:', answer_start, answer_end)

            # get input ids as list
            input_id_list = input_ids.squeeze().tolist()

            answer = self.ai.tokenizer.convert_tokens_to_string(
                self.ai.tokenizer.convert_ids_to_tokens(
                    input_id_list[answer_start:answer_end]
                )
            )

            # print(f"Question: {question}")
            # print(f"Answer: {answer}")

            gen_answers.append(answer)

        return SimpleNamespace(answers=gen_answers)
