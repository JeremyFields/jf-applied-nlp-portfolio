import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from tensorflow.keras.models import load_model

from measeval_quantity import (
    ExperimentConfig,
    evaluate_quantity,
    format_examples,
    get_vocabulary,
    integrate_sentences,
    load_data,
    make_predictions,
    predictions_to_labels,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference and evaluation")
    parser.add_argument("--data-dir", default="../data", help="Path to data directory")
    parser.add_argument("--model-path", required=True, help="Path to .keras model")
    parser.add_argument("--batch-size", type=int, default=64, help="Prediction batch size")
    parser.add_argument("--save-csv", default="../artifacts/test_predictions.csv", help="Where to save test predictions")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ExperimentConfig(batch_size=args.batch_size)
    data_dir = Path(args.data_dir)

    train_data = load_data(str(data_dir / "train.tsv"), False, cfg.seed)
    test_data = load_data(str(data_dir / "eval.tsv"), False, cfg.seed)

    _, word2idx, labels, label2idx = get_vocabulary(train_data)
    test_examples = integrate_sentences(test_data)
    x_test, _ = format_examples(test_examples, word2idx, label2idx, cfg.maxlen)

    model = load_model(args.model_path)
    predictions = make_predictions(model, x_test, cfg.batch_size)
    test_data = test_data.copy()
    test_data["prediction"] = predictions_to_labels(predictions, x_test, labels)

    report = evaluate_quantity(test_data)
    print(report)

    out_path = Path(args.save_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(test_data).to_csv(out_path, index=False)
    print(f"Saved predictions to {out_path}")


if __name__ == "__main__":
    main()
