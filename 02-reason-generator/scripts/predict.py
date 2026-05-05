from pathlib import Path
import argparse
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling import load_model


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a commonsense explanation for one statement.")
    parser.add_argument("statement", help="A nonsensical statement to explain.")
    parser.add_argument("--model-path", default="models/commonsense-reason-generator", help="Path to a fine-tuned checkpoint or a Hugging Face model name.")
    parser.add_argument("--max-length", type=int, default=25)
    parser.add_argument("--device", default="cpu", help="Model device, e.g. cpu or cuda.")
    return parser.parse_args()


def main():
    args = parse_args()
    model, tokenizer = load_model(args.model_path)
    model.to(args.device)
    model.eval()

    inputs = tokenizer(
        args.statement,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=args.max_length,
    ).to(args.device)

    generated_ids = model.generate(**inputs, max_length=args.max_length)
    prediction = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(prediction)


if __name__ == "__main__":
    main()