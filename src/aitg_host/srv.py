import time
import os
from aitg_host.util import get_compute_device
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
    # ensure req exists
    if not req:
        abort(400, "key not provided") # bad
    # ensure key provided
    key_id = "key"
    if key_id not in req:
        abort(401, "key not provided")
    # ensure key correct
    req_key = req[key_id]
    if req_key != API_KEY:
        abort(401)

@route("/info", method=["GET"])
def info_route():
    req_json = request.json
    try:
        verify_key(req_json)
    except KeyError as ke:
        abort(400, f"missing field {ke}")

    global AI_INSTANCE

    return json.dumps({"model": AI_INSTANCE.model_name})

@route("/encode", method=["GET", "POST"])
def encode_route():
    req_json = request.json
    try:
        verify_key(req_json)
        text = req_json["text"]
    except KeyError as ke:
        abort(400, f"missing field {ke}")

    global GENERATOR
    tokens = GENERATOR.str_to_toks(text)

    return json.dumps({"tokens": tokens})


@route("/decode", method=["GET", "POST"])
def decode_route():
    req_json = request.json
    try:
        verify_key(req_json)
        tokens = req_json["tokens"]
    except KeyError as ke:
        abort(400, f"missing field {ke}")

    global GENERATOR
    text = GENERATOR.toks_to_str(tokens)

    return json.dumps({"text": text})


@route("/gen", method=["GET", "POST"])
def gen_route():
    req_json = request.json
    try:
        verify_key(req_json)
        prompt = req_json["prompt"]
    except KeyError as ke:
        abort(400, f"missing field {ke}")

    # get params
    # mode params
    opt_use_rounds: bool = get_req_opt(req_json, "use_rounds", False)
    opt_max_rounds: int = get_req_opt(req_json, "max_rounds", 4)
    # option params
    opt_context_amount: float = get_req_opt(req_json, "context_amount", 0.5)
    opt_temp: float = get_req_opt(req_json, "temp", 0.9)
    opt_max_length: int = get_req_opt(req_json, "max_length", 256)
    opt_min_length: int = get_req_opt(req_json, "min_length", 0)
    opt_seed: int = get_req_opt(req_json, "seed", None)
    opt_top_p: float = get_req_opt(req_json, "top_p", 0.9)
    opt_top_k: int = get_req_opt(req_json, "top_k", 0)
    opt_repetition_penalty: float = get_req_opt(req_json, "repetition_penalty", 1.0)
    opt_length_penalty: float = get_req_opt(req_json, "length_penalty", 1.0)
    opt_no_repeat_ngram_size: int = get_req_opt(req_json, "no_repeat_ngram_size", 0)

    logger.debug(f"requesting generation for prompt: {prompt}")

    # generate
    try:
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
        gen_txt = AI_INSTANCE.filter_text(gen_txt)
        gen_txt_size = len(gen_txt)
        logger.debug(f"model output: {gen_txt}")
        generation_time = time.time() - start
        total_gen_num = len(gen_toks)
        gen_tps = num_new / generation_time
        logger.info(
            f"generated [{num_new}/{total_gen_num}] ({generation_time:.2f}s/{(gen_tps):.2f}tps)"
        )

        # success
        response.headers["Content-Type"] = "application/json"
        return json.dumps(
            {
                "text": gen_txt,
                "text_length": gen_txt_size,
                "text_tokens": gen_toks,
                "text_token_count": total_gen_num,
                "gen_new": num_new,
                "gen_total": total_gen_num,
                "gen_time": generation_time,
                "gen_tps": gen_tps,
            }
        )
    except Exception as ex:
        logger.error(f"error generating: {ex}")
        abort(400, f"generation failed")


def prepare_model(optimize: bool):
    start = time.time()
    logger.info(f"initializing[{get_compute_device()}]...")
    from aitg_host.model import load_model
    logger.info(f"init in: {time.time() - start:.2f}s")
    start = time.time()
    logger.info("loading model...")
    ai = load_model(MODEL_DIR, optimize)
    logger.info(f"finished loading in: {time.time() - start:.2f}s")
    logger.info(f"model: {ai.model_name}")

    return ai


def server(
    host: str = "localhost",
    port: int = 6000,
    debug: bool = False,
    optimize: bool = True,
):
    global AI_INSTANCE, GENERATOR
    AI_INSTANCE = ai = prepare_model(optimize)
    GENERATOR = SlidingGenerator(ai)

    logger.info(f"starting server on {host}:{port}")
    run(host=host, port=port, debug=debug)


def main():
    typer.run(server)


if __name__ == "__main__":
    main()
