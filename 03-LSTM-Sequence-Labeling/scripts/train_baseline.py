import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from tensorflow.keras.utils import set_random_seed

from measeval_quantity import (
    ExperimentConfig,
    create_model,
    format_examples,
    get_vocabulary,
    integrate_sentences,
    load_data,
    train_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline sequence labeling model")
    parser.add_argument("--data-dir", default="../data", help="Path to data directory")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ExperimentConfig(
        epochs=args.epochs or ExperimentConfig().epochs,
        batch_size=args.batch_size or ExperimentConfig().batch_size,
    )

    set_random_seed(cfg.seed)
    data_dir = Path(args.data_dir)

    train_data = load_data(str(data_dir / "train.tsv"), cfg.shrink_dataset, cfg.seed)
    dev_data = load_data(str(data_dir / "trial.tsv"), cfg.shrink_dataset, cfg.seed)

    vocab, word2idx, labels, label2idx = get_vocabulary(train_data)
    train_examples = integrate_sentences(train_data)
    dev_examples = integrate_sentences(dev_data)

    x_train, y_train = format_examples(train_examples, word2idx, label2idx, cfg.maxlen)
    x_dev, y_dev = format_examples(dev_examples, word2idx, label2idx, cfg.maxlen)

    model = create_model(len(vocab), len(labels), cfg.maxlen, cfg.embedding_dim, cfg.rnn_units)
    train_model(model, x_train, y_train, x_dev, y_dev, cfg.batch_size, cfg.epochs)

    output_dir = Path("../artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(output_dir / "baseline_model.keras")
    print(f"Saved model to {output_dir / 'baseline_model.keras'}")


if __name__ == "__main__":
    main()
