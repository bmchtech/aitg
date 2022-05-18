import os
import sys
import requests

AI_URL = None
TEMP = 0.2
SAMPLE_LEN = 32

def multiline_in(prompt=''):
    print(prompt, end='')
    sys.stdout.flush()
    return sys.stdin.read()

def complete_code(context, tries=1):
    global AI_URL, TEMP, SAMPLE_LEN
    # print(f'using prompt: {context}')

    # print(' --- sending request ---')

    res = requests.post(
        AI_URL + '/gen_sfcodegen.json',
        json={
            "context": context,
            "max_length": 2048,
            "sample_length": SAMPLE_LEN,
            "temperature": TEMP,
            "num_seqs": tries,
        }
    )

    # get response as json
    res_json = res.json()

    # print(f' --- received response ---')

    # print generated text
    # print(res_json['texts'][0])

    return res_json['texts']

def main():
    global AI_URL, TEMP, SAMPLE_LEN
    AI_URL = sys.argv[1]
    
    SAMPLE_LEN = int(sys.argv[2])
    TEMP = float(sys.argv[3])

    # let's try doing interactive literate programming

    code_so_far = ''

    # ask the user for some input
    USER_PROMPT = '>>> '
    user_input = multiline_in(USER_PROMPT) 
    while user_input:
        # print(f'user input: {user_input}')

        # generate code
        code_so_far += user_input
        # code_so_far += '\n'

        user_accepted_code = False
        selected_gen_code = ''

        while not user_accepted_code:
            print(' --- waiting for ai ---')
            generated_code = complete_code(code_so_far, tries=1)
            # print(f'generated code: {generated_code}')
            print(' --- generated code ---')
            print(generated_code[0])

            # ask user to accept or reject the generated code
            user_resp = input('accept generated code? [y/n] ').strip()

            if user_resp == 'y':
                user_accepted_code = True
                selected_gen_code = generated_code[0]
            if user_resp == 'e':
                # let user write their own code
                selected_gen_code = multiline_in('\ne>> ')
                user_accepted_code = True

        # user accepted the generated code
        # add the generated code to the code so far
        code_so_far += selected_gen_code

        print(' --- code so far ---')
        print(code_so_far)

        # ask the user for some input
        user_input = multiline_in(USER_PROMPT)

if __name__ == '__main__':
    main()