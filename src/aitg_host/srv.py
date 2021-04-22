import time
import os
from aitg_host.util import multiline_in
import typer

from bottle import run, route, request, response, abort
import json
from loguru import logger

MODEL_DIR = os.environ["MODEL"]
API_KEY = os.environ["KEY"]

AI_INSTANCE = None

def get_req_opt(req, name, default):
    if name in req:
        return req[name]
    else:
        return default

@route('/gen', method=['GET', 'POST'])
def gen_route():
    req_json = request.json

    try:
        # required stuff
        req_key = req_json['key']
        if req_key != API_KEY:
            abort(401)
        prompt = req_json['prompt']
    except KeyError as ke:
        abort(400, f'missing field {ke}')

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

    logger.debug(f'requesting generation for prompt: {prompt}')

    # generate
    start = time.time()
    global AI_INSTANCE
    gen_txt = AI_INSTANCE.generate_one(
        max_length=opt_max_length,
        prompt=prompt,
        temperature=opt_temp,
        top_p=opt_top_p,
        top_k=opt_top_k,
        repetition_penalty=opt_repetition_penalty,
        length_penalty=opt_length_penalty,
        no_repeat_ngram_size=opt_no_repeat_ngram_size,
    )
    gen_txt_size = len(gen_txt)
    logger.debug(f'model output: {gen_txt}')
    logger.info(f"generated {gen_txt_size} chars in: {time.time() - start:.2f}s")

    # success
    response.headers['Content-Type'] = 'application/json'
    return json.dumps(
        {
            'text': gen_txt,
            'text_length': gen_txt_size,
        }
    )

def prepare_model(optimize: bool):
    start = time.time()
    logger.info("initializing...")
    from aitg_host.model import load_model

    logger.info(f"init in: {time.time() - start:.2f}s")
    start = time.time()
    logger.info("loading model...")
    ai = load_model(MODEL_DIR, optimize)
    logger.info(f"finished loading in: {time.time() - start:.2f}s")

    return ai

def server(
    host:str = 'localhost',
    port:int = 6000,
    debug: bool = False,
    optimize: bool = True,
):
    global AI_INSTANCE
    AI_INSTANCE = prepare_model(optimize)

    logger.info(f'starting server on {host}:{port}')
    run(host=host, port=port, debug=debug)

def main():
    typer.run(server)


if __name__ == "__main__":
    main()
