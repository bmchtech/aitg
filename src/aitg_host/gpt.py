import logging
import os
import platform
import re
import shutil
import sys
from datetime import datetime
from random import randint
from typing import List, Optional, Union
from types import SimpleNamespace

import torch
import numpy as np
import torch.nn.functional as F
from pkg_resources import resource_filename
from tqdm.auto import trange
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    PreTrainedTokenizerFast,
)
from transformers.models.gpt2.convert_gpt2_original_tf_checkpoint_to_pytorch import (
    convert_gpt2_checkpoint_to_pytorch,
)

from aitextgen.TokenDataset import TokenDataset
from aitextgen.utils import (
    download_gpt2,
    find_index_of_subset,
    model_max_length,
    reset_seed,
    set_seed,
)
import aitextgen

def generate(
    ai,
    prompt: str = "",
    prepend_bos: bool = None,
    min_length: int = None,
    max_length: int = 256,
    temperature: float = 0.7,
    do_sample: bool = True,
    # return_as_list: bool = False,
    seed: int = None,
    pad_token_id: str = None,
    # schema: str = False,
    # normalize_key: bool = True,
    use_cache: bool = True,
    lstrip: bool = True,
    skip_special_tokens: bool = True,
    **kwargs,
) -> Optional[str]:
    prompt_text = prompt
    prompt_tensors = ai.tokenizer(text=prompt, return_tensors="pt")

    prompt_num_tokens = prompt_tensors.input_ids.size()[1]
    assert prompt_num_tokens < model_max_length(
        ai.model.config
    ), f"The prompt is too large for the model. ({prompt_num_tokens} tokens)"

    input_ids = prompt_tensors.input_ids.to(ai.get_device()) if prompt else None

    if prepend_bos is None:
        prepend_bos = getattr(ai.model.config, "line_by_line", None)

    if prepend_bos:
        bos = torch.tensor([[ai.tokenizer.bos_token_id]]).to(ai.get_device())
        if prompt:
            input_ids = torch.cat((bos, input_ids), dim=1)
        else:
            input_ids = bos

    if seed:
        set_seed(seed)

    if pad_token_id is None:
        pad_token_id = getattr(ai.tokenizer, "pad_token_id", None) or getattr(
            ai.tokenizer, "eos_token_id", None
        )

    # prevent an error from using a length greater than the model
    gen_max_length = model_max_length(ai.model.config)
    max_length = min(gen_max_length, max_length)

    while True:
        model_output = ai.model.generate(
            input_ids=input_ids,
            min_length=min_length,
            max_length=max_length,
            temperature=temperature,
            do_sample=do_sample,
            num_return_sequences=1,
            pad_token_id=pad_token_id,
            use_cache=use_cache,
            # for rich info
            return_dict_in_generate=True,
            output_scores=True,
            # misc args
            **kwargs,
        )

        # manual decode the sequences
        gen_texts = []
        gen_tokens = []
        gen_seqs = []
        # since num_return_sequences=1, we KNOW there is only 1 sequence
        for seq in model_output.sequences:
            gen_seqs.append(seq.tolist()) # output sequence

            # convert whole sequence to string
            decoded_sequence_text = ai.tokenizer.decode(
                seq, skip_special_tokens=skip_special_tokens
            )
            gen_texts.append(decoded_sequence_text)
            # print(f'decoded: {seq} -> {decoded_sequence}')

            # convert token by token
            filtered_tokens = ai.tokenizer.convert_ids_to_tokens(
                seq, skip_special_tokens=skip_special_tokens
            )
            gen_tokens.append(filtered_tokens)
            # strred_tokens = ai.tokenizer.convert_tokens_to_string(filtered_tokens)
            # print(f'convtok: {filtered_tokens} -> {strred_tokens}')

        # Handle stripping tokenization spaces w/ regex
        if lstrip:
            gen_texts = [re.sub(r"^\s+", "", text) for text in gen_texts]

        # if there is no generated text after cleanup, try again.
        if len(gen_texts) == 0:
            continue

        # now process the scores
        # let's stack the logits generated at each step to a tensor
        logits = torch.stack(model_output.scores, dim=1)
        # transform logits to softmax probs
        probs = logits.softmax(-1)

        # handle empty prompts, which really are bos_token under the hood
        # if prompt_num_tokens == 0:
        #     # we need padding for our probs
        #     # pad axis 1, for bos, to make the token count match the sequence
        #     probs = F.pad(probs, pad=(0, 0, 0, 1), value=0)
        
        # print("probs", probs.size(), probs)

        # # print probability pairings
        # num_new_tokens = len(gen_tokens[0]) - prompt_num_tokens
        # print(f"probs: {len(probs[0])}, toks: {num_new_tokens}")
        # for step_ix in range(num_new_tokens):
        #     tok_ix = prompt_num_tokens + step_ix # token index
        #     chosen_tok = gen_tokens[0][tok_ix] # chosen token str
        #     chosen_tok_id = gen_seqs[0][tok_ix] # chosen token id
        #     step_probs = probs[0, step_ix, :] # list of probs at this step
        #     chosen_tok_prob = step_probs[chosen_tok_id] # prob of chosen token
        #     print(f" prob[{step_ix:03}]: {chosen_tok:<20} | {chosen_tok_prob:<20}")

        # Reset seed if used
        if seed:
            reset_seed()

        
        # return processed results

        return SimpleNamespace(
            text = gen_texts[0],
            tokens = gen_tokens[0],
            seq = gen_seqs[0],
            prompt_ids = prompt_tensors.input_ids.tolist()[0],
            probs = probs[0, :, :].tolist()
        )
