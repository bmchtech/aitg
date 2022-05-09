
# internals of aitg

## enhancements to huggingface models

- friendly name can be put in config json, in `model_friendly_id`
- filter function can be put in `filter.py`, in the form: `filter_text(text: str) -> str`
