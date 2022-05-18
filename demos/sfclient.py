import os
import sys
import requests

def main():
    url = sys.argv[1]
    prompt_file = sys.argv[2]
    max_length = int(sys.argv[3])
    sample_length = int(sys.argv[4])
    temp = float(sys.argv[5])

    with open(prompt_file, 'r') as f:
        prompt_text = f.read()

    # do http request
    # printf '{                                                                                                                                                                                                                                  ─╯
    #         "context": "    '\'''\'''\'' Compute the nth Fibonacci number. '\'''\'''\''",
    #         "max_length": 64,
    #         "max_length_sample": 64
    # }' | http POST 'http://localhost:6000/gen_sfcodegen.json' | jq -r '.texts[0]'

    print(f'using prompt: {prompt_text}')

    print(' --- sending request ---')

    res = requests.post(
        url + '/gen_sfcodegen.json',
        json={
            "context": prompt_text,
            "max_length": max_length,
            "max_length_sample": sample_length,
            "temperature": temp,
        }
    )

    # get response as json
    res_json = res.json()

    print(f' --- received response ---')

    # print generated text
    print(res_json['texts'][0])

if __name__ == '__main__':
    main()