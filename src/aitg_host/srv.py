import time
import os
import traceback

import typer
from typing import List, Dict
from bottle import run, route, request, response, abort
from loguru import logger
import json
import msgpack
from aitg_host.models.question_answer import QuestionAnswerAI
import lz4.frame

from aitg_host import __version__, ICON_ART
from aitg_host.util import get_compute_device

# generators
import aitg_host.model
from aitg_host.gens.sliding_generator import SlidingGenerator
from aitg_host.gens.bart_summary_generator import BartSummaryGenerator
from aitg_host.gens.led_summary_generator import LedSummaryGenerator
from aitg_host.gens.classifier_generator import ClassifierGenerator
from aitg_host.gens.embed_generator import EmbedGenerator
from aitg_host.gens.qa_generator import QuestionAnswerGenerator
from aitg_host.gens.t5_generator import T5Generator

MODEL_TYPE = None
MODEL_DIR = os.environ.get("MODEL")
API_KEY = os.environ.get("KEY")
AITG_SRV_HOST = os.environ.get("AITG_SRV_HOST")
AITG_SRV_PORT = os.environ.get("AITG_SRV_PORT")

AI_INSTANCE = None
GENERATOR = None


def req_as_dict(req):
    try:
        if request.json:
            return request.json
        elif request.params:
            return request.params
        else:
            return None
    except:
        abort(400, f"invalid request json")


def get_req_opt(req, name, default):
    if name in req:
        return req[name]
    else:
        return default


def verify_req(req):
    # ensure req exists
    if not req:
        abort(400, "request is empty")  # bad
    # if key is specified, verify
    if API_KEY:
        # ensure key provided
        key_id = "key"
        if key_id not in req:
            abort(401, "key not provided")
        # ensure key correct
        req_key = req[key_id]
        if req_key != API_KEY:
            abort(401, "key is invalid")


def pack_bundle(bundle, ext):
    if ext == "json":
        response.headers["Content-Type"] = "application/json"
        return json.dumps(bundle)
    elif ext == "mp":
        response.headers["Content-Type"] = "application/x-msgpack"
        return msgpack.dumps(bundle)
    elif ext == "mpz":
        response.headers["Content-Type"] = "application/octet-stream"
        return lz4.frame.compress(msgpack.dumps(bundle))
    else:  # default
        return None


def ensure_model_type(model_type):
    global MODEL_TYPE
    if MODEL_TYPE != model_type:
        raise RuntimeError(
            f"model type mismatch! expected {model_type}, but got {MODEL_TYPE}"
        )


@route("/info.<ext>", method=["GET"])
def info_route(ext):
    req_json = req_as_dict(request)
    try:
        verify_req(req_json)
    except KeyError as ke:
        abort(400, f"missing field {ke}")

    global AI_INSTANCE

    bundle = {
        "server": "aitg_host",
        "version": __version__,
        "model": AI_INSTANCE.model_name,
        "model_type": AI_INSTANCE.model_type,
    }

    return pack_bundle(bundle, ext)


@route("/encode.<ext>", method=["GET", "POST"])
def encode_route(ext):
    req_json = req_as_dict(request)
    try:
        verify_req(req_json)
        text = req_json["text"]
    except KeyError as ke:
        abort(400, f"missing field {ke}")

    global GENERATOR
    tokens = GENERATOR.str_to_toks(text)

    return pack_bundle({"num_tokens": len(tokens), "tokens": tokens}, ext)


@route("/decode.<ext>", method=["GET", "POST"])
def decode_route(ext):
    req_json = req_as_dict(request)
    try:
        verify_req(req_json)
        tokens = req_json["tokens"]
    except KeyError as ke:
        abort(400, f"missing field {ke}")

    global GENERATOR
    text = GENERATOR.toks_to_str(tokens)

    return pack_bundle({"text": text}, ext)


@route("/gen_gpt.<ext>", method=["GET", "POST"])
def gen_gpt_route(ext):
    req_json = req_as_dict(request)
    try:
        verify_req(req_json)
        _ = req_json["prompt"]
    except KeyError as ke:
        abort(400, f"missing field {ke}")

    # mode params
    opt_include_probs: bool = get_req_opt(req_json, "include_probs", False)
    # option params
    opt_prompt: str = get_req_opt(req_json, "prompt", "")
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

        global AI_INSTANCE, GENERATOR, MODEL_TYPE
        ensure_model_type("gpt")

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
        logger.error(f"error generating: {traceback.format_exc()}")
        abort(400, f"generation failed")


@route("/gen_bart_summarizer.<ext>", method=["GET", "POST"])
def gen_bart_summarizer_route(ext):
    req_json = req_as_dict(request)
    try:
        verify_req(req_json)
        _ = req_json["text"]
    except KeyError as ke:
        abort(400, f"missing field {ke}")

    # mode params
    # option params
    opt_text: str = get_req_opt(req_json, "text", "")
    opt_max_length: int = get_req_opt(req_json, "max_length", 256)
    opt_min_length: int = get_req_opt(req_json, "min_length", 0)
    opt_num_beams: int = get_req_opt(req_json, "num_beams", 6)
    opt_repetition_penalty: float = get_req_opt(req_json, "repetition_penalty", 1.0)
    opt_length_penalty: float = get_req_opt(req_json, "length_penalty", 1.0)
    opt_max_time: float = get_req_opt(req_json, "opt_max_time", None)
    opt_no_repeat_ngram_size: int = get_req_opt(req_json, "no_repeat_ngram_size", 3)

    logger.debug(f"requesting generation for text: {opt_text}")

    # generate
    try:
        start = time.time()

        global AI_INSTANCE, GENERATOR, MODEL_TYPE
        ensure_model_type("bart_summarizer")

        # standard generate
        output = GENERATOR.generate(
            article=opt_text,
            max_length=opt_max_length,
            min_length=opt_min_length,
            num_beams=opt_num_beams,
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
            f"generated [{prompt_token_count}->{output.num_new}] ({generation_time:.2f}s/{(gen_tps):.2f}tps)"
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

        return pack_bundle(resp_bundle, ext)
    except Exception as ex:
        logger.error(f"error generating: {traceback.format_exc()}")
        abort(400, f"generation failed")


@route("/gen_led_summarizer.<ext>", method=["GET", "POST"])
def gen_led_summarizer_route(ext):
    req_json = req_as_dict(request)
    try:
        verify_req(req_json)
        _ = req_json["text"]
    except KeyError as ke:
        abort(400, f"missing field {ke}")

    # mode params
    opt_include_tokens: bool = get_req_opt(req_json, "include_tokens", False)
    # option params
    opt_text: str = get_req_opt(req_json, "text", "")
    opt_max_length: int = get_req_opt(req_json, "max_length", 512)
    opt_min_length: int = get_req_opt(req_json, "min_length", 0)
    opt_num_beams: int = get_req_opt(req_json, "num_beams", 6)
    opt_repetition_penalty: float = get_req_opt(req_json, "repetition_penalty", 1.0)
    opt_length_penalty: float = get_req_opt(req_json, "length_penalty", 1.8)
    opt_max_time: float = get_req_opt(req_json, "opt_max_time", None)
    opt_no_repeat_ngram_size: int = get_req_opt(req_json, "no_repeat_ngram_size", 3)

    logger.debug(f"requesting generation for text: {opt_text}")

    # generate
    try:
        start = time.time()

        global AI_INSTANCE, GENERATOR, MODEL_TYPE
        ensure_model_type("led_summarizer")

        # standard generate
        output = GENERATOR.generate(
            article=opt_text,
            max_length=opt_max_length,
            min_length=opt_min_length,
            num_beams=opt_num_beams,
            repetition_penalty=opt_repetition_penalty,
            length_penalty=opt_length_penalty,
            max_time=opt_max_time,
            no_repeat_ngram_size=opt_no_repeat_ngram_size,
        )

        logger.debug(f"model output: {output.text}")
        gen_txt = AI_INSTANCE.filter_text(output.text)
        gen_txt_size = len(gen_txt)
        total_gen_num = len(output.tokens)
        prompt_token_count = output.num_prompt_tokens
        generation_time = time.time() - start
        gen_tps = output.num_new / generation_time
        logger.info(
            f"generated [{prompt_token_count}->{output.num_new}] ({generation_time:.2f}s/{(gen_tps):.2f}tps)"
        )

        # done generating, now return the results over http

        # create base response bundle
        resp_bundle = {
            "text": gen_txt,
            "text_length": gen_txt_size,
            "prompt_token_count": prompt_token_count,
            "num_new": output.num_new,
            "token_count": total_gen_num,
            "gen_time": generation_time,
            "gen_tps": gen_tps,
            "model": AI_INSTANCE.model_name,
        }

        # add optional sections
        if opt_include_tokens:
            resp_bundle["tokens"] = output.tokens

        return pack_bundle(resp_bundle, ext)
    except Exception as ex:
        logger.error(f"error generating: {traceback.format_exc()}")
        abort(400, f"generation failed")


@route("/gen_bart_classifier.<ext>", method=["GET", "POST"])
def gen_bart_classifier_route(ext):
    req_json = req_as_dict(request)
    try:
        verify_req(req_json)
        _ = req_json["text"]
        _ = req_json["classes"]
    except KeyError as ke:
        abort(400, f"missing field {ke}")

    # mode params
    opt_include_probs: bool = get_req_opt(req_json, "include_probs", False)
    # option params
    opt_text: str = get_req_opt(req_json, "text", None)
    opt_classes: List[str] = get_req_opt(req_json, "classes", None)
    opt_hypothesis_template: str = get_req_opt(
        req_json, "hypothesis_template", "This example is {}."
    )
    opt_multi_label: bool = get_req_opt(req_json, "multi_label", False)

    logger.debug(
        f"requesting classification for text: {opt_text}, classes: {opt_classes}"
    )

    # generate
    try:
        start = time.time()

        global AI_INSTANCE, GENERATOR, MODEL_TYPE
        ensure_model_type("bart_classifier")

        # standard generate
        output = GENERATOR.generate(
            text=opt_text,
            candidate_labels=opt_classes,
            hypothesis_template=opt_hypothesis_template,
            multi_label=opt_multi_label,
        )

        gen_score_pairs = list(zip(output.labels, output.scores))
        num_classes = len(opt_classes)
        logger.debug(f"model output: {gen_score_pairs}")
        generation_time = time.time() - start
        logger.info(f"generated [{num_classes} cls] ({generation_time:.2f}s)")

        # done generating, now return the results over http

        # create base response bundle
        resp_bundle = {
            "labels": output.labels,
            "scores": output.scores,
            "gen_time": generation_time,
            "model": AI_INSTANCE.model_name,
        }

        return pack_bundle(resp_bundle, ext)
    except Exception as ex:
        logger.error(f"error generating: {traceback.format_exc()}")
        abort(400, f"generation failed")


@route("/gen_sentence_embed.<ext>", method=["GET", "POST"])
def gen_bart_classifier_route(ext):
    req_json = req_as_dict(request)
    try:
        verify_req(req_json)
        _ = req_json["texts"]
    except KeyError as ke:
        abort(400, f"missing field {ke}")

    # mode params
    # option params
    opt_texts: List[str] = get_req_opt(req_json, "texts", None)

    logger.debug(f"requesting sentence embeds for texts: {opt_texts}")

    # generate
    try:
        start = time.time()

        global AI_INSTANCE, GENERATOR, MODEL_TYPE
        ensure_model_type("sentence_embed")

        # standard generate
        output = GENERATOR.generate(
            texts=opt_texts,
        )

        embeds = output.embeddings
        num_embeds = len(embeds)
        logger.debug(f"model output: embeds[{len(embeds)}]")
        generation_time = time.time() - start
        gen_vecps = num_embeds / generation_time
        logger.info(
            f"generated [{num_embeds} vec] ({generation_time:.2f}s/{gen_vecps:.2f} vps)"
        )

        # done generating, now return the results over http

        # create base response bundle
        resp_bundle = {
            "similarity": output.similarity,
            "num_embeds": num_embeds,
            "embeds": embeds,
            "gen_time": generation_time,
            "model": AI_INSTANCE.model_name,
        }

        return pack_bundle(resp_bundle, ext)
    except Exception as ex:
        logger.error(f"error generating: {traceback.format_exc()}")
        abort(400, f"generation failed")


@route("/gen_question_answer.<ext>", method=["GET", "POST"])
def gen_question_answer_route(ext):
    req_json = req_as_dict(request)
    try:
        verify_req(req_json)
        _ = req_json["text"]
        _ = req_json["questions"]
    except KeyError as ke:
        abort(400, f"missing field {ke}")

    # mode params
    # option params
    opt_text: str = get_req_opt(req_json, "text", None)
    opt_questions: List[str] = get_req_opt(req_json, "questions", None)

    logger.debug(
        f"requesting question answers for text: {opt_text}, questions: {opt_questions}"
    )

    # generate
    try:
        start = time.time()

        global AI_INSTANCE, GENERATOR, MODEL_TYPE
        ensure_model_type("question_answer")

        # standard generate
        output = GENERATOR.generate(
            text=opt_text,
            questions=opt_questions,
        )

        gen_answers = output.answers
        num_answers = len(gen_answers)
        logger.debug(f"model output: {gen_answers}")
        generation_time = time.time() - start
        gen_ansps = num_answers / generation_time
        logger.info(
            f"generated [{num_answers} ans] ({generation_time:.2f}s/{gen_ansps:.2f} aps)"
        )

        # done generating, now return the results over http

        # create base response bundle
        resp_bundle = {
            "answers": gen_answers,
            "scores": output.scores,
            "gen_time": generation_time,
            "model": AI_INSTANCE.model_name,
        }

        return pack_bundle(resp_bundle, ext)
    except Exception as ex:
        logger.error(f"error generating: {traceback.format_exc()}")
        abort(400, f"generation failed")


@route("/gen_t5.<ext>", method=["GET", "POST"])
def gen_t5_route(ext):
    req_json = req_as_dict(request)
    try:
        verify_req(req_json)
        _ = req_json["text"]
    except KeyError as ke:
        abort(400, f"missing field {ke}")

    # mode params
    # option params
    opt_text: str = get_req_opt(req_json, "text", "")
    opt_max_length: int = get_req_opt(req_json, "max_length", 256)
    opt_min_length: int = get_req_opt(req_json, "min_length", 0)
    opt_max_time: float = get_req_opt(req_json, "opt_max_time", None)

    logger.debug(f"requesting generation for text: {opt_text}")

    # generate
    try:
        start = time.time()

        global AI_INSTANCE, GENERATOR, MODEL_TYPE
        ensure_model_type("t5")

        # standard generate
        output = GENERATOR.generate(
            text=opt_text,
            max_length=opt_max_length,
            min_length=opt_min_length,
            max_time=opt_max_time,
        )

        gen_txt = AI_INSTANCE.filter_text(output.text)
        gen_txt_size = len(gen_txt)
        prompt_token_count = len(output.prompt_ids)
        logger.debug(f"model output: {gen_txt}")
        generation_time = time.time() - start
        total_gen_num = len(output.tokens)
        gen_tps = output.num_new / generation_time
        logger.info(
            f"generated [{prompt_token_count}->{output.num_new}] ({generation_time:.2f}s/{(gen_tps):.2f}tps)"
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

        return pack_bundle(resp_bundle, ext)
    except Exception as ex:
        logger.error(f"error generating: {traceback.format_exc()}")
        abort(400, f"generation failed")

def prepare_model(load_model_func):
    # sanity checks
    if not MODEL_DIR:
        raise RuntimeError(
            "no model specified. please pass a path to your model in the MODEL environment variable"
        )

    # load
    start = time.time()
    logger.info("loading model...")
    ai = load_model_func(MODEL_DIR)
    logger.info(f"finished loading in: {time.time() - start:.2f}s")
    logger.info(f"model: {ai.model_name} ({ai.model_type})")

    return ai


def server(
    model_type: str,
    host: str = "localhost",
    port: int = 6000,
    debug: bool = False,
):
    # first init
    start = time.time()
    print("\n\n", ICON_ART, f"\n            AITG HOST v{__version__}\n\n")
    # logger.info(f"aitg_host server v{__version__}")
    logger.info(f"initializing[{get_compute_device()[1]}]...")
    logger.info(f"init in: {time.time() - start:.2f}s")

    # select model type
    load_func = None
    generator_func = lambda ai: None
    if model_type == "gpt":
        load_func = aitg_host.model.load_gpt_model
        generator_func = lambda ai: SlidingGenerator(ai)
    elif model_type == "bart_summarizer":
        load_func = aitg_host.model.load_bart_summarizer_model
        generator_func = lambda ai: BartSummaryGenerator(ai)
    elif model_type == "led_summarizer":
        load_func = aitg_host.model.load_led_summarizer_model
        generator_func = lambda ai: LedSummaryGenerator(ai)
    elif model_type == "bart_classifier":
        load_func = aitg_host.model.load_bart_classifier_model
        generator_func = lambda ai: ClassifierGenerator(ai)
    elif model_type == "sentence_embed":
        load_func = aitg_host.model.load_sentence_embed_model
        generator_func = lambda ai: EmbedGenerator(ai)
    elif model_type == "question_answer":
        load_func = aitg_host.model.load_question_answer_model
        generator_func = lambda ai: QuestionAnswerGenerator(ai)
    elif model_type == "t5":
        load_func = aitg_host.model.load_t5_model
        generator_func = lambda ai: T5Generator(ai)
    else:
        # unknown
        raise RuntimeError(f"unknown model_type: {model_type}")

    global AI_INSTANCE, GENERATOR, MODEL_TYPE
    MODEL_TYPE = model_type
    AI_INSTANCE = ai = prepare_model(load_func)
    GENERATOR = generator_func(ai)

    # environ overrides to CLI
    global AITG_SRV_HOST, AITG_SRV_PORT
    if AITG_SRV_HOST:
        host = AITG_SRV_HOST
    if AITG_SRV_PORT:
        port = int(AITG_SRV_PORT)

    logger.info(f"starting server on {host}:{port}")
    run(host=host, port=port, debug=debug)


def main():
    typer.run(server)


if __name__ == "__main__":
    main()
