import torch
import torch.nn.functional as F
from aitg.gens.base import BaseGenerator
from types import SimpleNamespace
from typing import List


class QuestionAnswerGenerator(BaseGenerator):
    def __init__(self, ai):
        super().__init__(ai)

    def str_to_ids(self, text):
        # custom tokenizer invocation
        return self.ai.tokenizer(text=text, add_special_tokens=True).input_ids

    def generate(self, text: str, questions: List[str], lstrip: bool = True, **kwargs):
        gen_answers = []
        gen_answer_probs = []
        for question in questions:
            # encode
            input_tensor = self.ai.tokenizer(
                question, text, add_special_tokens=True, return_tensors="pt"
            )
            input_ids = input_tensor.input_ids.to(self.ai.device)

            with torch.no_grad():
                outputs = self.ai.model(input_ids, **kwargs)
            answer_start_scores = outputs.start_logits
            answer_end_scores = outputs.end_logits

            answer_start_probs = answer_start_scores.softmax(-1).squeeze()
            answer_end_probs = answer_end_scores.softmax(-1).squeeze()

            # get the most likely beginning/end of answer with the argmax of the score
            answer_start_ix = torch.argmax(answer_start_scores)
            answer_end_ix = torch.argmax(answer_end_scores)

            answer_start = answer_start_ix
            answer_end = answer_end_ix + 1

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

            answer_start_prob_max = answer_start_probs[answer_start_ix].numpy().item()
            answer_end_prob_max = answer_end_probs[answer_end_ix].numpy().item()

            # answer_probs = [answer_start_prob_max, answer_end_prob_max]
            answer_probs = answer_start_prob_max * answer_end_prob_max

            # print('answer probs:', answer_probs)

            if lstrip:
                answer = self.lstrip_texts([answer])[0]

            gen_answers.append(answer)
            gen_answer_probs.append(answer_probs)

        return SimpleNamespace(answers=gen_answers, scores=gen_answer_probs)
