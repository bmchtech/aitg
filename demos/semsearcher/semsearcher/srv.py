import time
import os
import typer

from bottle import run, route, request, response, static_file, abort
import json
from loguru import logger

DEBUG = os.environ.get('DEBUG')

def get_req_opt(req, name, default):
    if name in req:
        return req[name]
    else:
        return default

# static path for ui
@route('/ui')
@route('/ui/<filepath:path>')
def serve_ui(filepath='index.html'):
    return static_file(filepath, root='ui')

# search route with query
@route('/search/<query>', method='GET')
def req_search(query):
    req_json = request.json
    # opt_test = get_req_opt(req_json, 'test', False)

    logger.debug(f'searching for: {query}')

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
    run(host=host, port=port, debug=DEBUG)

def main():
    typer.run(server)

if __name__ == "__main__":
    main()