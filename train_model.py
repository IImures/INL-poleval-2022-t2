from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.baseline import build_majority_dictionary, predict_rows
from src.data_io import load_labeled_split
from src.metrics import calculate_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train baseline abbreviation disambiguation model (majority dictionary)."
    )
    parser.add_argument("--train-dir", default="dataset/train", help="Path to train split directory")
    parser.add_argument("--dev-dir", default="dataset/dev-0", help="Path to dev split directory")
    parser.add_argument("--model-dir", default="models", help="Output directory for saved artifacts")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    train_rows = load_labeled_split(args.train_dir)
    majority_dictionary = build_majority_dictionary(train_rows)

    majority_path = model_dir / "majority_dictionary.json"
    with majority_path.open("w", encoding="utf-8") as f:
        json.dump(majority_dictionary, f, ensure_ascii=False, indent=2)

    metadata = {
        "model_type": "majority_dictionary_baseline",
        "train_dir": args.train_dir,
        "dev_dir": args.dev_dir,
        "train_rows": len(train_rows),
        "dictionary_size": len(majority_dictionary),
        "score_formula": "0.25 * Af + 0.75 * Ab",
        "evaluation": "case-insensitive exact match",
    }

    dev_expected_path = Path(args.dev_dir) / "expected.tsv"
    if dev_expected_path.exists():
        dev_rows = load_labeled_split(args.dev_dir)
        dev_predictions = predict_rows(dev_rows, majority_dictionary)

        true_expanded = [row["expanded"] for row in dev_rows]
        true_base = [row["base"] for row in dev_rows]
        pred_expanded = [row["expanded"] for row in dev_predictions]
        pred_base = [row["base"] for row in dev_predictions]

        scores = calculate_scores(
            true_expanded=true_expanded,
            pred_expanded=pred_expanded,
            true_base=true_base,
            pred_base=pred_base,
        )

        metadata["dev_rows"] = len(dev_rows)
        metadata["dev_scores"] = scores

        print(f"Expanded accuracy Af: {scores['Af']:.4f}")
        print(f"Base accuracy Ab: {scores['Ab']:.4f}")
        print(f"Final score: {scores['Final']:.4f}")
    else:
        print(f"No expected.tsv in dev dir: {args.dev_dir}. Skipping evaluation.")

    metadata_path = model_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Saved majority dictionary to: {majority_path}")
    print(f"Saved metadata to: {metadata_path}")


if __name__ == "__main__":
    main()

