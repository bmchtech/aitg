from transformers import AutoTokenizer, AutoModelForCausalLM
from optimum.onnxruntime import ORTModelForCausalLM
from optimum.onnxruntime import ORTOptimizer
from optimum.onnxruntime.configuration import OptimizationConfig
import torch
import typer
import os
import sys

def cli(
    model_path: str,
    feature: str,
    out_path: str,
    opt: int = 1,
):
    print(f"loading model from {model_path}")
    optimizer = ORTOptimizer.from_pretrained(model_path, feature=feature)

    # optimize and save
    print(f"running optimization and saving to {out_path}")
    os.mkdir(out_path)
    optimizer.export(
        out_path + '/model.onnx',
        out_path + '/model_optimized.onnx',
        OptimizationConfig(optimization_level=opt)
    )

def main():
    typer.run(cli)

if __name__ == "__main__":
    main()
