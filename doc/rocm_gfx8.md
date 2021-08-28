go to inference
conda install -y packaging==20.9
pip install .
go to host / src
pip install typer colorama bottle loguru
MODEL=/projects/Temp/PT_GPTNEO125_ATG python -m aitg_host.cli
MODEL=/projects/Temp/PT_GPTNEO125_ATG KEY=secret python -m aitg_host.srv gpt --host 0.0.0.0