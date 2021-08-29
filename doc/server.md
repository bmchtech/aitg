# aitg_host server

## intro

this provides a lightweight and plug-and-play http server component for utilizing gpt models!

## run server

first you want to run the server. if you want to use docker see the [docs for docker](docker.md).
otherwise, you want to run the `aitg_host.srv` module.

for example:
```sh
KEY=secret MODEL=/tmp/PT_GPTNEO125_ATG python -m aitg_host.srv gpt --host 127.0.0.1 --port 6000
```

## api

a very simple api that provides various methods.
- either GET or POST can be used
- file extension on the request path specifies the return data type
    - `.json` = JSON
    - `.mp` = MsgPack
    - `.mpz` = LZ4(MsgPack)
- request body is always json data

### routes
- `info` - get info about the server and model
- `encode` - tokenize text to string
- `decode` - turn a token sequence to a string
- `gen_gpt` - generate text

### try on your shell!

give it a try with curl:
```sh
curl --request GET \
  --url http://localhost:6000/gen_gpt.json \
  --header 'Content-Type: application/json' \
  --data '{
        "key": "secret",
        "prompt": "I like elephants because",
        "max_length": 24
}'
```