# based on https://github.com/python-poetry/poetry/issues/1879#issuecomment-592133519

# `python-base` sets up all our shared environment variables
FROM python:3.9.1-alpine as python-base

    # python
ENV PYTHONUNBUFFERED=1 \
    # prevents python creating .pyc files
    PYTHONDONTWRITEBYTECODE=1 \
    \
    # pip
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    \
    # poetry
    # make poetry create the virtual environment in the project's root
    # it gets named `.venv`
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    # do not ask any interactive question
    POETRY_NO_INTERACTION=1 \
    \
    # paths
    # this is where our requirements + virtual environment will live
    PYSETUP_PATH="/opt/pysetup" \
    VENV_PATH="/opt/pysetup/.venv"


# prepend poetry and venv to path
ENV PATH="$POETRY_HOME/bin:$VENV_PATH/bin:$PATH"


# `builder-base` stage is used to build deps + create our virtual environment
FROM python-base as builder-base
RUN apk update \
    && apk add --no-cache \
        # deps for installing poetry
        curl \
        # deps for building python deps
        alpine-sdk libffi-dev py3-cryptography

# python deps
RUN pip install poetry

# install poetry - respects $POETRY_VERSION & $POETRY_HOME
RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -

# copy project requirement files here to ensure they will be cached.
WORKDIR $PYSETUP_PATH
COPY poetry.lock pyproject.toml ./

# install runtime deps - uses $POETRY_VIRTUALENVS_IN_PROJECT internally
# RUN poetry install --no-dev
RUN poetry export -f requirements.txt > requirements.txt

# `development` image is used during development / testing
FROM python-base as development
ENV FASTAPI_ENV=development
WORKDIR $PYSETUP_PATH

# quicker install as runtime deps are already installed
# RUN poetry install
RUN pip install -r requirements.txt

# copy stuff
COPY ./aitg_host /app/aitg_host

# vars
ENV MODEL=/app/model

WORKDIR /app
RUN "ls"
CMD [ "python", "-m", "aitg_host.cli" ]