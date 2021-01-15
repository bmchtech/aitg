
# ai text generator host

## setup

enter `src/`

```sh
poetry install
```

## run

cli:
```sh
MODEL=/path/to/your_model poetry run aitg_host_cli
```

server:
```sh
MODEL=/path/to/your_model KEY=secret poetry run aitg_host_srv
```

see [instructions](doc/docker.md) for running in Docker.