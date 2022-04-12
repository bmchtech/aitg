
# multi

just commands for running various tasks

## tasks

```sh
MODEL=~/Downloads/PT_GPTNEO125_ATG poetry run aitg_host_srv gpt

MODEL=~/Downloads/PT_DistilBART_CNN_SSHLEIFER_1206 poetry run aitg_host_srv bart_summarizer

MODEL=~/Downloads/PT_DistilBART_MNLI_VALHALLA_1209 poetry run aitg_host_srv bart_classifier

MODEL=~/Downloads/PT_MPNet_PARAPHRASE_v2 poetry run aitg_host_srv sentence_embed

MODEL=~/Downloads/PT_MiniLM_UNCASED_SQUAD2 poetry run aitg_host_srv question_answer 

MODEL=~/Downloads/PT_T5S_BASE poetry run aitg_host_srv t5
```