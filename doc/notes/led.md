# led pubmed model

currently, i've tested this with:
`PT_LED_LARGE16384_PUBMED`

here are various recommendations

## params
- max length: 128-512, above this it sometimes gets hairy
- no_repeat_ngram_size = 3
- early_stopping = true
- num_beams = 6
- length_penalty = 1.8

## filter

`filter.py`:
```py
import re

def filter_text(text):
    text = re.sub(r'(doi:10.*)', ' ', text)
    text = re.sub(r'(see research article.*)', ' ', text)
    text = re.sub(r'(\(?\[\s?1\s?\] \[\s?2\s?\] \[\s?3\s?\] \[\s?4\s?\].*)', ' ', text)
    text = re.sub(r'(\[\s?2\s?\] \[\s?3\s?\] \[\s?4\s?\].*)', ' ', text)
    text = re.sub(r'(imagesfigure.*)', ' ', text)
    text = re.sub(r'(figure 1.figure 2.*)', ' ', text)
    text = re.sub(r'([Tt]he authors \[1\].{0,3}[Tt]he authors \[2\].*)', ' ', text)
    text = re.sub(r'(see the full text.*)', ' ', text)
    text = re.sub(r'([Pp]ublished by)?\s{0,3}([Jj]ohn [Ww]iley & [Ss]ons ltd.*)', ' ', text)
    text = re.sub(r'([Hh]ttp:\/\/[Ww]{0,3}\.?[Bb]iomed.*)', ' ', text)
    text = re.sub(r'(http:\/\/j\.thomson.*)', ' ', text)
    
    return text

```