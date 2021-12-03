import time
import os
import typer

from bottle import run, route, request, response, static_file, abort
import json
from loguru import logger

from semsearcher.cli import multisearch_index_cmd

DEBUG = os.environ.get("DEBUG")
AITG_SERVER = ""
SEARCH_DIR = ""


def get_req_opt(req, name, default):
    if not req:
        return default
    if name in req:
        return req[name]
    else:
        return default


# static path for ui
@route("/ui")
@route("/ui/<filepath:path>")
def serve_ui(filepath="index.html"):
    return static_file(filepath, root="ui")


# search route with query
@route("/search/<query>", method=["GET", "POST"])
def req_search(query):
    req_json = request.json
    # opt_test = get_req_opt(req_json, 'test', False)
    opt_num_results = int(get_req_opt(req_json, "num_results", 16))

    logger.debug(f"searching for: {query}")

    # generate
    start = time.time()

    # search
    global AITG_SERVER, SEARCH_DIR
    results = multisearch_index_cmd(AITG_SERVER, SEARCH_DIR, query, n=opt_num_results)

    # response
    response.content_type = "application/json"

    total_time = time.time() - start
    num_results = 0
    logger.info(f"generated {num_results} results in: {total_time}s")

    dict_results = []
    for (doc_name, score, sent, context) in results:
        dict_results.append(
            {
                "doc": doc_name,
                "score": score,
                "sent": sent,
                "context": context,
            }
        )

    # success
    response.headers["Content-Type"] = "application/json"
    return json.dumps(
        {
            "results": dict_results,
            "time": total_time,
        }
    )


def server(
    server: str,
    in_indexes_dir: str,
    host: str = "localhost",
    port: int = 8442,
):
    global AITG_SERVER, SEARCH_DIR
    AITG_SERVER = server
    SEARCH_DIR = in_indexes_dir

    logger.info(f"starting server on {host}:{port}")
    run(host=host, port=port, debug=DEBUG)


def main():
    typer.run(server)


if __name__ == "__main__":
    main()
