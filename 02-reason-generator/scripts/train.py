from pathlib import Path
import sys
import argparse

import pandas as pd
from transformers import enable_full_determinism

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TrainingConfig
from src.data import load_data, prepare_datasets
from src.inference import evaluate_prediction, make_predictions
from src.modeling import load_model
from src.training import create_trainer, create_training_arguments


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune BART to generate commonsense explanations.")
    parser.add_argument("--train-data", default="SemEval2020-Task4-Data/ALL data/Training  Data/subtaskC_data_all.csv")
    parser.add_argument("--train-answers", default="SemEval2020-Task4-Data/ALL data/Training  Data/subtaskC_answers_all.csv")
    parser.add_argument("--dev-data", default="SemEval2020-Task4-Data/ALL data/Dev Data/subtaskC_dev_data.csv")
    parser.add_argument("--dev-answers", default="SemEval2020-Task4-Data/ALL data/Dev Data/subtaskC_gold_answers.csv")
    parser.add_argument("--test-data", default="SemEval2020-Task4-Data/ALL data/Test Data/subtaskC_test_data.csv")
    parser.add_argument("--test-answers", default="SemEval2020-Task4-Data/ALL data/Test Data/subtaskC_gold_answers.csv")
    parser.add_argument("--model-name", default=TrainingConfig.model_name)
    parser.add_argument("--max-length", type=int, default=TrainingConfig.max_length)
    parser.add_argument("--epochs", type=int, default=TrainingConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainingConfig.train_batch_size)
    parser.add_argument("--learning-rate", type=float, default=TrainingConfig.learning_rate)
    parser.add_argument("--output-dir", default=TrainingConfig.output_dir)
    parser.add_argument("--metric", choices=["bleu", "rouge"], default="rouge")
    return parser.parse_args()


def main():
    args = parse_args()
    enable_full_determinism(seed=TrainingConfig.seed)

    model, tokenizer = load_model(args.model_name)

    train_data = load_data(args.train_data, args.train_answers)
    dev_data = load_data(args.dev_data, args.dev_answers)
    test_data = load_data(args.test_data, args.test_answers, is_test=True)

    train_dataset, dev_dataset, test_dataset = prepare_datasets(
        train_data, dev_data, test_data, tokenizer, args.max_length
    )

    train_args = create_training_arguments(
        args.epochs,
        args.batch_size,
        args.learning_rate,
        args.output_dir,
        args.max_length,
    )
    trainer = create_trainer(model, train_args, train_dataset, dev_dataset, tokenizer)
    trainer.train()

    predictions = make_predictions(trainer, test_dataset, tokenizer)
    test_data = test_data.copy()
    test_data["prediction"] = predictions

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print(evaluate_prediction(test_data, args.metric))


if __name__ == "__main__":
    main()
