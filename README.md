
# ai text generator host

## setup

### host

first, you need to install all the dependencies to run the gpt2 host (using the `aitextgen` library with the `pytorch` backend)

enter `src/`

```sh
poetry install
```

### model

next, you need to grab a pytorch model.
for more info on how to train/fine-tune, see [my blog post](https://blog.rie.icu/post/microfinetuning_gpt2/).

if you don't have your own yet, you can use a sample model:
+ [DistilGPT2-S Base](https://github.com/xdrie/aitextgen_host/releases/download/v1.0.0/PT_DistilGPT2_ATG.7z)
+ [GPT2-S Philosophical Babbler](https://github.com/xdrie/aitextgen_host/releases/download/v1.0.0/PhilBabble_ATG_20201201_071644__snap6k.7z)

a model directory should contain `config.json` and `pytorch_model.bin`.
## run

finally, point your host to a model and run.

cli:
```sh
MODEL=/path/to/your_model poetry run aitg_host_cli
```

server:
```sh
MODEL=/path/to/your_model KEY=secret poetry run aitg_host_srv
```

by default, the quantization optimization is applied to the model on initialization. this generally sacrifices a bit of accuracy, while providing about a 20% speedup. to disable, pass `--no-optimize`.

see [instructions](doc/docker.md) for running in Docker.

### cli tips
press Ctrl+D (sometimes twice) to send an EOF after entering your prompt, and then the model will generate text.
