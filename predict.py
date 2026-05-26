from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.baseline import predict_rows
from src.data_io import load_unlabeled_input, write_predictions_tsv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict abbreviation expansions using saved majority baseline."
    )
    parser.add_argument("--input", required=True, help="Path to test/input in.tsv file")
    parser.add_argument("--model-dir", default="models", help="Directory with trained model artifacts")
    parser.add_argument("--output", required=True, help="Path to output predictions TSV")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    majority_path = model_dir / "majority_dictionary.json"

    if not majority_path.exists():
        raise FileNotFoundError(
            f"Missing majority dictionary at {majority_path}. Run train_model.py first."
        )

    with majority_path.open("r", encoding="utf-8") as f:
        majority_dictionary = json.load(f)

    input_rows = load_unlabeled_input(args.input)
    predictions = predict_rows(input_rows, majority_dictionary)
    write_predictions_tsv(predictions, args.output)

    print(f"Loaded rows: {len(input_rows)}")
    print(f"Saved predictions to: {args.output}")


if __name__ == "__main__":
    main()

