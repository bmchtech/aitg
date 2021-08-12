import time
import os
from aitg_host.util import compute_device
import typer

from bottle import run, route, request, response, abort
import json
from loguru import logger

from aitg_host.sliding_generator import SlidingGenerator

MODEL_DIR = os.environ["MODEL"]
API_KEY = os.environ["KEY"]

AI_INSTANCE = None
GENERATOR = None

def get_req_opt(req, name, default):
    if name in req:
        return req[name]
    else:
        return default

def verify_key(req):
    req_key = req['key']
    if req_key != API_KEY:
        abort(401)

@route('/encode', method=['GET', 'POST'])
def encode_route():
    req_json = request.json
    try:
        verify_key(req_json)
        text = req_json['text']
    except KeyError as ke:
        abort(400, f'missing field {ke}')

    global GENERATOR
    tokens = GENERATOR.str_to_toks(text)

    return json.dumps(
        {
            'tokens': tokens
        }
    )

@route('/decode', method=['GET', 'POST'])
def decode_route():
    req_json = request.json
    try:
        verify_key(req_json)
        tokens = req_json['tokens']
    except KeyError as ke:
        abort(400, f'missing field {ke}')

    global GENERATOR
    text = GENERATOR.toks_to_str(tokens)

    return json.dumps(
        {
            'text': text
        }
    )

@route('/gen', method=['GET', 'POST'])
def gen_route():
    req_json = request.json
    try:
        verify_key(req_json)
        prompt = req_json['prompt']
    except KeyError as ke:
        abort(400, f'missing field {ke}')

    # get params
    # mode params
    opt_use_rounds: bool = get_req_opt(req_json, 'use_rounds', False)
    opt_max_rounds: int = get_req_opt(req_json, 'max_rounds', 4)
    # option params
    opt_context_amount: float = get_req_opt(req_json, 'context_amount', 0.5)
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
    
    global AI_INSTANCE, GENERATOR

    if opt_use_rounds:
        gen_txt, gen_toks = GENERATOR.generate_rounds(
            prompt=prompt,
            max_rounds=opt_max_rounds,
            context_amount=opt_context_amount,
            temperature=opt_temp,
            max_length=opt_max_length,
            min_length=opt_min_length,
            seed=opt_seed,
            top_p=opt_top_p,
            top_k=opt_top_k,
            repetition_penalty=opt_repetition_penalty,
            length_penalty=opt_length_penalty,
            no_repeat_ngram_size=opt_no_repeat_ngram_size,
        )
        num_new = 0
    else:
        # standard generate
        gen_txt, gen_toks, num_new = GENERATOR.generate(
            prompt=prompt,
            fresh=True,
            temperature=opt_temp,
            max_length=opt_max_length,
            min_length=opt_min_length,
            seed=opt_seed,
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
            'text_tokens': gen_toks,
            'text_token_count': len(gen_toks),
            'text_new': num_new,
        }
    )

def prepare_model(optimize: bool):
    start = time.time()
    print(Style.NORMAL + Fore.CYAN + f"initializing[{compute_device()}]...")
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
    global AI_INSTANCE, GENERATOR
    AI_INSTANCE = ai = prepare_model(optimize)
    GENERATOR = SlidingGenerator(ai)

    logger.info(f'starting server on {host}:{port}')
    run(host=host, port=port, debug=debug)

def main():
    typer.run(server)


if __name__ == "__main__":
    main()
