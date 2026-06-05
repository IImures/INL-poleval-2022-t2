from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from src.baseline import predict_rows as predict_rows_with_majority
from src.data_io import load_unlabeled_input, write_predictions_tsv
from src.ml_model import predict_with_ml_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict abbreviation expansions using saved majority baseline."
    )
    parser.add_argument("--input", required=True, help="Path to test/input in.tsv file")
    parser.add_argument("--model-dir", default="models", help="Directory with trained model artifacts")
    parser.add_argument("--output", required=True, help="Path to output predictions TSV")
    parser.add_argument(
        "--method",
        choices=["ml", "majority"],
        default="ml",
        help="Prediction method: TF-IDF ML model or majority baseline",
    )
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

    if args.method == "majority":
        predictions = predict_rows_with_majority(input_rows, majority_dictionary)
    else:
        expanded_model_path = model_dir / "expanded_model.joblib"
        base_model_path = model_dir / "base_model.joblib"

        if not expanded_model_path.exists() or not base_model_path.exists():
            raise FileNotFoundError(
                "Missing TF-IDF model artifacts. Run train_model.py first, "
                "or use --method majority."
            )

        expanded_model = joblib.load(expanded_model_path)
        base_model = joblib.load(base_model_path)
        predictions = predict_with_ml_models(
            rows=input_rows,
            expanded_model=expanded_model,
            base_model=base_model,
        )

    write_predictions_tsv(predictions, args.output)

    print(f"Loaded rows: {len(input_rows)}")
    print(f"Prediction method: {args.method}")
    print(f"Saved predictions to: {args.output}")


if __name__ == "__main__":
    main()

