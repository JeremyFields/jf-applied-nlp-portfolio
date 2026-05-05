# Commonsense Reason Generator

A compact NLP project that fine-tunes BART to generate a short commonsense explanation for a nonsensical statement.

This project is based on SemEval-2020 ComVE Subtask C.

## What it does

- loads the ComVE statement/reason CSV files
- tokenizes statement-reason pairs for sequence-to-sequence training
- fine-tunes a BART model with Hugging Face Transformers
- decodes generated explanations during inference
- evaluates outputs with BLEU or ROUGE
- includes a tiny CLI for generating one explanation from a single statement

## Project layout

- `src/data.py` - data loading, preprocessing, and dataset preparation
- `src/modeling.py` - model and tokenizer loading
- `src/training.py` - training arguments and Trainer setup
- `src/inference.py` - generation and evaluation helpers
- `scripts/train.py` - end-to-end training entry point
- `scripts/predict.py` - one-off inference for a single statement

## Setup

```bash
pip install -r requirements.txt
```

## Data

Download the SemEval-2020 ComVE files and place them under `SemEval2020-Task4-Data/`

```bash
git clone https://github.com/wangcunxiang/SemEval2020-Task4-Commonsense-Validation-and-Explanation.git SemEval2020-Task4-Data
```

The project does not store dataset files in git.

## Run training

```bash
python scripts/train.py --metric rouge
```

Example with a different model size or training config:

```bash
python scripts/train.py --model-name facebook/bart-base --epochs 3 --batch-size 8 --learning-rate 1e-5 --max-length 25
```

## Quick Inference

Generate one explanation from a statement:

```bash
python scripts/predict.py "He put an elephant into the fridge."
```

If you trained a model, point the script at the saved checkpoint:

```bash
python scripts/predict.py "He put an elephant into the fridge." --model-path models/commonsense-reason-generator
```

## Notes

- The project keeps the setup lightweight on purpose.
- It is intended to show the end-to-end flow for a seq2seq NLP task.
- If you want a stronger demo, train with more data and keep the same pipeline.
