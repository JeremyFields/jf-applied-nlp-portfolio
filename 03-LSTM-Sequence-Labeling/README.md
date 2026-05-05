  # Quantity Extraction with BiLSTM (MeasEval)

Project for sequence labeling of quantities in scientific text.

## Highlights

  - Modular NLP pipeline (preprocessing, training, inference, evaluation)
  - Baseline BiLSTM sequence tagger and GloVe-initialized variant
  - Reproducible scripts for training and evaluation
  - Clear data provenance and licensing guidance for portfolio use

## Project Structure

```text
portfolio/
  pyproject.toml
  requirements.txt
  README.md
  scripts/
    train_baseline.py
    train_with_glove.py
    evaluate_model.py
  src/
    measeval_quantity/
      __init__.py
      config.py
      data.py
      preprocess.py
      model.py
      embeddings.py
      train.py
      predict.py
      evaluate.py
```

## Setup

From the `portfolio` folder:

```bash
pip install -r requirements.txt
pip install -e .
```

## Data Layout

This project expects the dataset in `../data` by default:

- `../data/train.tsv`
- `../data/trial.tsv`
- `../data/eval.tsv`

Expected columns:

- `docId`
- `sentId`
- `lemma`
- `label`

## Data Provenance and Access

This repository does not redistribute the original assignment dataset files.

The code was developed against course-provided TSV files in the MeasEval quantity-labeling format. Since the redistribution license for the course copy is unclear, dataset files are intentionally excluded from this project.

To reproduce training and evaluation, obtain the dataset from official MeasEval sources and place files in the paths shown above.

References:

- SemEval 2021 Task 8 (MeasEval) CodaLab page: https://competitions.codalab.org/competitions/25770
- MeasEval GitHub repository (linked from CodaLab): https://github.com/harperco/MeasEval

Reproducibility note:

- This repository is code-reproducible.
- Full data reproducibility depends on obtaining the official data distribution under its original terms.

GloVe download:

```bash
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip -d glove glove.6B.zip
```

GloVe path default:

- `../glove.6B/glove.6B.300d.txt`

## Train Baseline

```bash
python scripts/train_baseline.py --data-dir ../data --epochs 5 --batch-size 64
```

Model artifact:

- `../artifacts/baseline_model.keras`

## Train with GloVe Initialization

```bash
python scripts/train_with_glove.py --data-dir ../data --glove-path ../glove.6B/glove.6B.300d.txt
```

Model artifact:

- `../artifacts/glove_model.keras`

## Evaluate a Saved Model

```bash
python scripts/evaluate_model.py --model-path ../artifacts/glove_model.keras --data-dir ../data
```

Outputs:

- Prints classification report for `Quantity`
- Saves token-level predictions to `../artifacts/test_predictions.csv`

## Notes

- Embedding layer in the GloVe model is trainable (`trainable=True`).
- Padding index is `0` and masked in the embedding layer (`mask_zero=True`).
