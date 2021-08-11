
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

finally, point your host to a model and run. to use huggingface, use `@` like this: `@EleutherAI/gpt-neo-2.7B`

cli:
```sh
MODEL=/path/to/your_model poetry run aitg_host_cli
```

server:
```sh
MODEL=/path/to/your_model KEY=secret poetry run aitg_host_srv
```

~~by default, the quantization optimization is applied to the model on initialization. this generally sacrifices a bit of accuracy, while providing about a 20% speedup. to disable, pass `--no-optimize`.~~

see [instructions](doc/docker.md) for running in Docker.

### cli tips
press Ctrl+D (sometimes twice) to send an EOF after entering your prompt, and then the model will generate text.

#### usage

```
Usage: aitg_host_cli [OPTIONS]

Options:
  --temp FLOAT                    [default: 0.9]
  --max-length INTEGER            [default: 256]
  --min-length INTEGER            [default: 0]
  --seed INTEGER
  --top-p FLOAT                   [default: 0.9]
  --top-k INTEGER                 [default: 0]
  --repetition-penalty FLOAT      [default: 1.0]
  --length-penalty FLOAT          [default: 1.0]
  --no-repeat-ngram-size INTEGER  [default: 0]
  --optimize / --no-optimize      [default: True]
  --install-completion [bash|zsh|fish|powershell|pwsh]
                                  Install completion for the specified shell.
  --show-completion [bash|zsh|fish|powershell|pwsh]
                                  Show completion for the specified shell, to
                                  copy it or customize the installation.

  --help                          Show this message and exit.
```

## server api

```json
{
    "key": "example",
    "prompt": "The quick brown",
    "temp": 0.9,
    "max_length": 256,
    "min_length": 0,
    "seed": null,
    "top_p": 0.9,
    "top_k": 0,
    "repetition_penalty": 1.0,
    "length_penalty": 1.0,
    "no_repeat_ngram_size": 0,
}
 # get params
    opt_temp: float = get_req_opt(req_json, 'temp', 0.9)
    opt_max_length: int = get_req_opt(req_json, 'max_length', 256)
    opt_min_length: int = get_req_opt(req_json, 'min_length', 0)
    opt_seed: int = get_req_opt(req_json, 'seed', None)
    opt_top_p: float = get_req_opt(req_json, 'top_p', 0.9)
    opt_top_k: int = get_req_opt(req_json, 'top_k', 0)
    opt_repetition_penalty: float = get_req_opt(req_json, 'repetition_penalty', 1.0)
    opt_length_penalty: float = get_req_opt(req_json, 'length_penalty', 1.0)
    opt_no_repeat_ngram_size: int = get_req_opt(req_json, 'no_repeat_ngram_size', 0)
```