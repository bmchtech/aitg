import os
import sys
import requests

server = os.environ["SERVER"]
IN_FILE = sys.argv[1]

contents = None
with open(IN_FILE) as f:
    contents = f.read()

resp = requests.post(
    server + "/gen_bart_summarizer.json",
    json={
        "text": contents,
        "num_beams": 4,
        "length_penalty": 2.0,
        "max_length": 256,
        "min_length": 128,
        "no_repeat_ngram_size": 3,
    },
)

print(resp.json()["text"])