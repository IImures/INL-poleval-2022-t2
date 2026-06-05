from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from src.baseline import build_candidate_dictionary, build_majority_dictionary, predict_rows
from src.data_io import load_labeled_split
from src.metrics import calculate_scores
from src.ml_model import predict_with_ml_models, split_prediction_columns, train_label_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train abbreviation disambiguation models: majority baseline + TF-IDF LinearSVC classifiers."
    )
    parser.add_argument("--train-dir", default="dataset/train", help="Path to train split directory")
    parser.add_argument("--dev-dir", default="dataset/dev-0", help="Path to dev split directory")
    parser.add_argument("--model-dir", default="models", help="Output directory for saved artifacts")
    return parser.parse_args()


def _score_predictions(rows: list[dict[str, str]], predictions: list[dict[str, str]]) -> dict[str, float]:
    true_expanded = [row["expanded"] for row in rows]
    true_base = [row["base"] for row in rows]
    pred_expanded, pred_base = split_prediction_columns(predictions)

    return calculate_scores(
        true_expanded=true_expanded,
        pred_expanded=pred_expanded,
        true_base=true_base,
        pred_base=pred_base,
    )


def _print_scores(title: str, scores: dict[str, float]) -> None:
    print(f"\n{title}:")
    print(f"  Expanded accuracy Af: {scores['Af']:.4f}")
    print(f"  Base accuracy Ab: {scores['Ab']:.4f}")
    print(f"  Final score: {scores['Final']:.4f}")


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    train_rows = load_labeled_split(args.train_dir)
    majority_dictionary = build_majority_dictionary(train_rows)
    candidate_dictionary = build_candidate_dictionary(train_rows)

    majority_path = model_dir / "majority_dictionary.json"
    with majority_path.open("w", encoding="utf-8") as f:
        json.dump(majority_dictionary, f, ensure_ascii=False, indent=2)

    candidate_path = model_dir / "candidate_dictionary.json"
    with candidate_path.open("w", encoding="utf-8") as f:
        json.dump(candidate_dictionary, f, ensure_ascii=False, indent=2)

    print("Training expanded-form TF-IDF classifier...")
    expanded_model = train_label_model(train_rows, "expanded")
    expanded_model_path = model_dir / "expanded_model.joblib"
    joblib.dump(expanded_model, expanded_model_path)

    print("Training base-form TF-IDF classifier...")
    base_model = train_label_model(train_rows, "base")
    base_model_path = model_dir / "base_model.joblib"
    joblib.dump(base_model, base_model_path)

    metadata = {
        "model_type": "tfidf_linear_svc_with_majority_baseline",
        "train_dir": args.train_dir,
        "dev_dir": args.dev_dir,
        "train_rows": len(train_rows),
        "dictionary_size": len(majority_dictionary),
        "features": ["word_tfidf_1_2", "char_wb_tfidf_2_5"],
        "classifier": "LinearSVC(max_iter=5000)",
        "score_formula": "0.25 * Af + 0.75 * Ab",
        "evaluation": "case-insensitive exact match",
    }

    dev_expected_path = Path(args.dev_dir) / "expected.tsv"
    if dev_expected_path.exists():
        dev_rows = load_labeled_split(args.dev_dir)

        majority_predictions = predict_rows(dev_rows, majority_dictionary)
        majority_scores = _score_predictions(dev_rows, majority_predictions)

        ml_predictions = predict_with_ml_models(
            rows=dev_rows,
            expanded_model=expanded_model,
            base_model=base_model,
            majority_dictionary=majority_dictionary,
            candidate_dictionary=candidate_dictionary,
        )
        ml_scores = _score_predictions(dev_rows, ml_predictions)

        metadata["dev_rows"] = len(dev_rows)
        metadata["dev_scores"] = {
            "majority_baseline": majority_scores,
            "tfidf_linear_svc": ml_scores,
        }

        _print_scores("Majority baseline", majority_scores)
        _print_scores("TF-IDF + LinearSVC", ml_scores)
    else:
        print(f"No expected.tsv in dev dir: {args.dev_dir}. Skipping evaluation.")

    metadata_path = model_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Saved majority dictionary to: {majority_path}")
    print(f"Saved candidate dictionary to: {candidate_path}")
    print(f"Saved expanded model to: {expanded_model_path}")
    print(f"Saved base model to: {base_model_path}")
    print(f"Saved metadata to: {metadata_path}")


if __name__ == "__main__":
    main()

