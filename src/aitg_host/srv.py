import time
import os
from aitg_host.util import get_compute_device
import typer

from bottle import run, route, request, response, abort
from loguru import logger
import json
import msgpack

from aitg_host.textgen.sliding_generator import SlidingGenerator

MODEL_DIR = os.environ["MODEL"]
API_KEY = os.environ["KEY"]

AI_INSTANCE = None
GENERATOR = None

def req_as_json(req):
    try:
        return request.json
    except:
        abort(400, f"invalid request json")

def get_req_opt(req, name, default):
    if name in req:
        return req[name]
    else:
        return default

def verify_key(req):
    # ensure req exists
    if not req:
        abort(400, "key not provided")  # bad
    # ensure key provided
    key_id = "key"
    if key_id not in req:
        abort(401, "key not provided")
    # ensure key correct
    req_key = req[key_id]
    if req_key != API_KEY:
        abort(401)

def pack_bundle(bundle, ext):
    if ext == 'json':
        response.headers["Content-Type"] = "application/json"
        return json.dumps(bundle)
    elif ext == 'mp':
        response.headers["Content-Type"] = "application/x-msgpack"
        return msgpack.dumps(bundle)
    elif ext == 'mpz':
        response.headers["Content-Type"] = "application/octet-stream"
        return msgpack.dumps(bundle)
    else: # default
        return None

@route("/info.<ext>", method=["GET"])
def info_route(ext):
    req_json = req_as_json(request)
    try:
        verify_key(req_json)
    except KeyError as ke:
        abort(400, f"missing field {ke}")

    global AI_INSTANCE

    return pack_bundle({"model": AI_INSTANCE.model_name}, ext)

@route("/encode.<ext>", method=["GET", "POST"])
def encode_route(ext):
    req_json = req_as_json(request)
    try:
        verify_key(req_json)
        text = req_json["text"]
    except KeyError as ke:
        abort(400, f"missing field {ke}")

    global GENERATOR
    tokens = GENERATOR.str_to_toks(text)

    return pack_bundle({"tokens": tokens}, ext)

@route("/decode.<ext>", method=["GET", "POST"])
def decode_route(ext):
    req_json = req_as_json(request)
    try:
        verify_key(req_json)
        tokens = req_json["tokens"]
    except KeyError as ke:
        abort(400, f"missing field {ke}")

    global GENERATOR
    text = GENERATOR.toks_to_str(tokens)

    return pack_bundle({"text": text}, ext)

@route("/gen.<ext>", method=["GET", "POST"])
def gen_route(ext):
    req_json = req_as_json(request)
    try:
        verify_key(req_json)
        _ = req_json["prompt"]
    except KeyError as ke:
        abort(400, f"missing field {ke}")

    # mode params
    opt_include_probs: bool = get_req_opt(req_json, "include_probs", False)
    # option params
    opt_prompt: float = get_req_opt(req_json, "prompt", "")
    opt_temp: float = get_req_opt(req_json, "temp", 0.9)
    opt_max_length: int = get_req_opt(req_json, "max_length", 256)
    opt_min_length: int = get_req_opt(req_json, "min_length", 0)
    opt_seed: int = get_req_opt(req_json, "seed", None)
    opt_top_p: float = get_req_opt(req_json, "top_p", 0.9)
    opt_top_k: int = get_req_opt(req_json, "top_k", 0)
    opt_repetition_penalty: float = get_req_opt(req_json, "repetition_penalty", 1.0)
    opt_length_penalty: float = get_req_opt(req_json, "length_penalty", 1.0)
    opt_max_time: float = get_req_opt(req_json, "opt_max_time", None)
    opt_no_repeat_ngram_size: int = get_req_opt(req_json, "no_repeat_ngram_size", 0)
    # lv2 params
    opt_flex_max_length: int = get_req_opt(req_json, "flex_max_length", 0)

    logger.debug(f"requesting generation for prompt: {opt_prompt}")

    # generate
    try:
        start = time.time()

        global AI_INSTANCE, GENERATOR

        # prompt
        prompt_tokens = tokens = GENERATOR.str_to_toks(opt_prompt)

        # apply lv2 params
        if opt_flex_max_length > 0:
            # we use chunked max length instead of the fixed value
            # find out how many full buckets the prompt uses
            prompt_num_buckets = len(prompt_tokens) // opt_flex_max_length
            auto_max_length = (prompt_num_buckets + 1) * opt_flex_max_length

            # ensure it's within the limit
            if opt_max_length > 0:
                max_length_limit = opt_max_length

            if auto_max_length > max_length_limit:
                opt_max_length = max_length_limit
            else:
                opt_max_length = auto_max_length

        # standard generate
        output = GENERATOR.generate(
            prompt=opt_prompt,
            temperature=opt_temp,
            max_length=opt_max_length,
            min_length=opt_min_length,
            seed=opt_seed,
            top_p=opt_top_p,
            top_k=opt_top_k,
            repetition_penalty=opt_repetition_penalty,
            length_penalty=opt_length_penalty,
            max_time=opt_max_time,
            no_repeat_ngram_size=opt_no_repeat_ngram_size,
        )

        gen_txt = AI_INSTANCE.filter_text(output.text)
        gen_txt_size = len(gen_txt)
        prompt_token_count = len(output.prompt_ids)
        logger.debug(f"model output: {gen_txt}")
        generation_time = time.time() - start
        total_gen_num = len(output.tokens)
        gen_tps = output.num_new / generation_time
        logger.info(
            f"generated [{output.num_new}/{total_gen_num}] ({generation_time:.2f}s/{(gen_tps):.2f}tps)"
        )

        # done generating, now return the results over http

        # create base response bundle
        resp_bundle = {
            "text": gen_txt,
            "text_length": gen_txt_size,
            "prompt_token_count": prompt_token_count,
            "tokens": output.tokens,
            "token_count": total_gen_num,
            "num_new": output.num_new,
            "num_total": total_gen_num,
            "gen_time": generation_time,
            "gen_tps": gen_tps,
            "model": AI_INSTANCE.model_name,
        }

        # add optional sections
        if opt_include_probs:
            resp_bundle["probs"] = output.probs

        return pack_bundle(resp_bundle, ext)
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
