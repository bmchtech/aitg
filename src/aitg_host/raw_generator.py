import logging
import os
import platform
import re
import shutil
import sys
from datetime import datetime
from random import randint
from typing import List, Optional, Union

import torch
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

def raw_generate(
        ai,
        prompt: str = "",
        prepend_bos: bool = None,
        min_length: int = None,
        max_length: int = 256,
        temperature: float = 0.7,
        do_sample: bool = True,
        return_as_list: bool = False,
        seed: int = None,
        pad_token_id: str = None,
        schema: str = False,
        normalize_key: bool = True,
        use_cache: bool = True,
        lstrip: bool = True,
        nonempty_output: bool = True,
        skip_special_tokens: bool = True,
        **kwargs,
    ) -> Optional[str]:
        prompt_text = prompt
        prompt_tensors = ai.tokenizer(text=prompt, return_tensors="pt")

        if prompt:
            prompt_num_tokens = list(prompt_tensors["input_ids"].shape)[1]
            assert prompt_num_tokens < model_max_length(
                ai.model.config
            ), f"The prompt is too large for the model. ({prompt_num_tokens} tokens)"

        input_ids = (
            prompt_tensors["input_ids"].to(ai.get_device()) if prompt else None
        )

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
            outputs = ai.model.generate(
                input_ids=input_ids,
                min_length=min_length,
                max_length=max_length,
                temperature=temperature,
                do_sample=do_sample,
                num_return_sequences=1,
                pad_token_id=pad_token_id,
                use_cache=use_cache,
                **kwargs,
            )
        
            # manual decode
            gen_texts = []
            for seq in outputs:
                decoded_sequence = ai.tokenizer.decode(seq, skip_special_tokens=skip_special_tokens)
                # print(f'decoded: {seq} -> {decoded_sequence}')
                # convert token by token
                filtered_tokens = ai.tokenizer.convert_ids_to_tokens(seq, skip_special_tokens=skip_special_tokens)
                strred_tokens = ai.tokenizer.convert_tokens_to_string(filtered_tokens)
                # print(f'convtok: {filtered_tokens} -> {strred_tokens}')
                gen_texts.append(decoded_sequence)

            # Handle stripping tokenization spaces w/ regex
            if lstrip:
                gen_texts = [re.sub(r"^\s+", "", text) for text in gen_texts]

            if nonempty_output:
                if min_length:
                    gen_texts = list(
                        filter(lambda x: len(x) > min_length, gen_texts)
                    )
                else:
                    gen_texts = list(filter(lambda x: len(x) > 0, gen_texts))

            # if there is no generated text after cleanup, try again.
            if len(gen_texts) == 0:
                continue

            # Reset seed if used
            if seed:
                reset_seed()

            # print('beep4')
            return gen_texts
