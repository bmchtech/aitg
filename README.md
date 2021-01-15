
# ai text generator host

## setup

put model in `models/` and update path to model

```sh
poetry install
```

## run

```sh
MODEL=/path/to/your_model poetry run aitg_host
```

## docker

build
```sh
docker build --pull -t aitg_host/cli -f Dockerfile .
```

run cli
```sh
docker run -it --rm -v $(pwd)/models/YOUR_MODEL:/app/model aitg_host/cli
```
