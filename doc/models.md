
# comparison of all released GPT models

| Model         | LAMBADA ppl | LAMBADA acc |
|---------------|-------------|-------------|
|               |             |             |
| GPT-2-117M    | 35.130      | 45.99%      |
| GPT-2-345M    | 15.600      | 55.48%      |
| GPT-2-762M    | 10.870      | 60.12%      |
| GPT-2-1542M   | 8.630       | 63.24%      |
|               |             |             |
| GPT-3-124M    | 18.600      | 42.70%      |
| GPT-3-350M    | 9.090       | 54.30%      |
| GPT-3-Ada     | 9.950       | 51.60%      |
| GPT-3-760M    | 6.530       | 60.40%      |
| GPT-3-1.3B    | 5.440       | 63.60%      |
| GPT-3-Babbage | 5.580       | 62.40%      |
| GPT-3-2.7B    | 4.600       | 67.10%      |
| GPT-3-6.7B    | 4.000       | 70.30%      |
| GPT-3-Curie   | 4.000       | 68.50%      |
| GPT-3-13B     | 3.560       | 72.50%      |
| GPT-3-175B    | 3.000       | 76.20%      |
| GPT-3-Davinci | 2.970       | 74.80%      |
|               |             |             |
| GPT-Neo-125M  | 30.266      | 37.36%      |
| GPT-Neo-350M  | 13.876      | 47.27%      |
| GPT-Neo-1.3B  | 7.498       | 57.23%      |
| GPT-Neo-2.7B  | 5.626       | 62.22%      |

## references

+ gpt2 numbers: [source](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
+ gpt3 numbers: [source](https://blog.eleuther.ai/gpt3-model-sizes/)
+ gptneo numbers for 1.3B and 2.7B: [source](https://github.com/EleutherAI/gpt-neo/#linguistic-reasoning)
+ gptneo numbers for 125M and 350M: my own testing using the [eleuther harness](https://github.com/EleutherAI/lm-evaluation-harness)