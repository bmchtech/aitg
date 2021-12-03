import time
import os
from aitg_host.util import multiline_in
import typer

from bottle import run, route, request, response, abort
import json
from loguru import logger

DEBUG = os.environ.get('DEBUG')

def get_req_opt(req, name, default):
    if name in req:
        return req[name]
    else:
        return default

@route('/gen', method='GET')
def gen_route():
    req_json = request.json
    opt_test = get_req_opt(req_json, 'test', False)
    opt_query = get_req_opt(req_json, 'query', '')

    logger.debug(f'searching for: {opt_query}')

    # generate
    start = time.time()
    total_time = time.time() - start
    num_results = 0
    logger.info(f"generated {num_results} results in: {total_time}s")

    # success
    response.headers['Content-Type'] = 'application/json'
    return json.dumps(
        {
            # 'text': gen_txt,
            # 'text_length': gen_txt_size,
        }
    )

def server(
    host:str = 'localhost',
    port:int = 8442,
):
    logger.info(f'starting server on {host}:{port}')

def main():
    typer.run(server)

if __name__ == "__main__":
    main()