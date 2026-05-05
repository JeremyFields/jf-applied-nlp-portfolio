import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from tensorflow.keras.utils import set_random_seed

from measeval_quantity import (
    ExperimentConfig,
    create_embedding_matrix,
    create_model_with_embeddings,
    format_examples,
    get_vocabulary,
    integrate_sentences,
    load_data,
    load_embeddings,
    train_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train sequence labeling model with GloVe")
    parser.add_argument("--data-dir", default="../data", help="Path to data directory")
    parser.add_argument("--glove-path", default="../glove.6B/glove.6B.300d.txt", help="Path to GloVe embeddings")
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

    embedding_index = load_embeddings(args.glove_path)
    embedding_matrix = create_embedding_matrix(
        embedding_index,
        word2idx,
        vocab_size=len(vocab),
        embedding_dim=cfg.embedding_dim,
    )

    model = create_model_with_embeddings(
        len(vocab), len(labels), cfg.maxlen, cfg.embedding_dim, cfg.rnn_units, embedding_matrix
    )
    train_model(model, x_train, y_train, x_dev, y_dev, cfg.batch_size, cfg.epochs)

    output_dir = Path("../artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(output_dir / "glove_model.keras")
    print(f"Saved model to {output_dir / 'glove_model.keras'}")


if __name__ == "__main__":
    main()
