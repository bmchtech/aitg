import torch
import torch.quantization
from aitg.util import get_compute_device

QUANTIZE_INT8 = 8
QUANTIZE_FLOAT16 = 16

def quantize_ai_model(ai, mode=QUANTIZE_INT8):
    # torch quantize

    if mode == QUANTIZE_INT8:
        quantized_model = torch.quantization.quantize_dynamic(ai.model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True)
    elif mode == QUANTIZE_FLOAT16:
        quantized_model = torch.quantization.quantize_dynamic(ai.model, {torch.nn.Linear}, dtype=torch.float16, inplace=True)

    return ai
