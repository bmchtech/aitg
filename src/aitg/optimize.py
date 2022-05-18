import torch
import torch.quantization
from aitg.util import get_compute_device

def quantize_ai_model(ai):
    # torch quantize

    quantized_model = torch.quantization.quantize_dynamic(ai.model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True)

    return ai
