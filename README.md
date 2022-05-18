
![icon](doc/icon.png)

# aitg

aitg ("ai text generator")

this project allows you to easily run many transformers models locally on the **command line** or as a **http server**

## quickstart

don't care about the details and just want to use GPT-2/GPT-3 super fast? this section is for you

grab model (`gpt3-125m`, 171 MB):
```sh
wget https://github.com/xdrie/aitextgen_host/releases/download/v1.5.2/PT_GPTNEO125_ATG.7z -O /tmp/PT_GPTNEO125_ATG.7z
7z x /tmp/PT_GPTNEO125_ATG.7z -o/tmp
```

run the container, cli:
```sh
docker run -it --rm -v /tmp/PT_GPTNEO125_ATG:/app/model xdrie/aitg:v1.6.0 aitg.cli
```

in the command line, press Ctrl+D (or whatever your eof key is) to submit a prompt.

## feature catalog

models:
- text generation (gpt)
- sequence to sequence generation (t5)
- code generation (sfcodegen)
- text summarization (bart-cnn, longformer-encoder-decoder)
- text classification (bart-mnli, minilm2-nli)
- text embedding (mpnet-paraphrase)
- question answering (minilm-squad2, roberta-squad2)

deployment:
- all of these models are available via http rest api
- optimized model execution with huggingface transformers
- gpu acceleration support

## run from source

### python project

first, you need to install all the dependencies to run the gpt2 host (using the `aitextgen` library with the `pytorch` backend)

enter `src/`

```sh
poetry install
```

### model

next, you need a pytorch model.

if you don't have your own yet, you can use a sample model:
+ [DistilGPT2-S Base](https://github.com/xdrie/aitextgen_host/releases/download/v1.0.0/PT_DistilGPT2_ATG.7z)

## cli usage

point your host to a model and run; usually this is a model directory with `config.json` and `pytorch_model.bin`. to use huggingface, use `@` like this: `@EleutherAI/gpt-neo-2.7B`

cli:
```sh
MODEL=/path/to/your_model poetry run aitg_cli
```

once it's done loading, it will ask you for a prompt. when you're done typing the prompt, press Ctrl+D (sometimes twice) to send an EOF after entering your prompt, and then the model will generate text.

```
Usage: aitg_cli [OPTIONS]

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

## server usage

run the server with:

```sh
MODEL=/path/to/your_model poetry run aitg_srv gpt
```

then

`GET /gen_gpt.json` with a JSON request body like the following:

```json
{
    "key": "secret",
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
```

that's the short version. see the [server docs](doc/server.md) for more details.

## other models

obviously, gpt-2/3 get the spotlight and were the original main focus of aitg. but now an array of other models and model types are supported.

more documentation on other models coming soon.

## docker usage

see [instructions](doc/docker.md) for running in Docker.