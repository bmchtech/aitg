
# docker instructions

## build images

build the cli image:
```sh
docker build --pull -t aitg -f docker/Dockerfile .

# optionally, save
docker save -o /tmp/aitg_$(git describe --long --tags | sed 's/^v//;s/\([^-]*-g\)/r\1/;s/-/./g')__docker.tar aitg
```

## run images

run cli:
```sh
docker run -it --rm -v $(pwd)/models/YOUR_MODEL:/app/model aitg aitg.cli
```

run server:
```sh
docker run -it --rm -v $(pwd)/models/YOUR_MODEL:/app/model -p 6000:6000 aitg aitg.srv gpt --host 0.0.0.0
```

test the server:
```sh
printf '{"key": "secret", "prompt": "%s", "max_length": %d}' "The quick brown" 16 | http GET localhost:6000/gen_gpt.json
```

## more examples

```sh
podman run -it --rm -v ~/Downloads/PT_DistilBART_MNLI_VALHALLA_1209:/app/model -p 6402:6000 docker.io/xdrie/aitg:v2.0.1 aitg.srv bart_classifier --host 0.0.0.0                                                                  
```
