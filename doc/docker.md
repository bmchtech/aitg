
# docker instructions

## build images

build the cli image:
```sh
docker build --pull -t aitg_host/cli -f docker/cli.Dockerfile .
```

## run images

run cli:
```sh
docker run -it --rm -v $(pwd)/models/YOUR_MODEL:/app/model aitg_host/cli
```
