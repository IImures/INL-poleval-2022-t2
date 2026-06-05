from __future__ import annotations

import argparse
from pathlib import Path

from src.baseline import build_majority_dictionary, predict_rows
from src.data_io import load_labeled_split
from src.metrics import calculate_scores
from src.ml_model import (
    build_vocab,
    describe_device,
    get_device,
    predict_with_ml_models,
    save_model_bundle,
    split_prediction_columns,
    train_label_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train abbreviation disambiguation models: majority baseline + PyTorch text classifiers."
    )
    parser.add_argument("--train-dir", default="dataset/train", help="Path to train split directory")
    parser.add_argument("--dev-dir", default="dataset/dev-0", help="Path to dev split directory")
    parser.add_argument("--model-dir", default="models", help="Output directory for saved artifacts")
    parser.add_argument("--device", default="auto", help="Torch device: auto, cuda, or cpu")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs per model")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size")
    parser.add_argument("--predict-batch-size", type=int, default=256, help="Evaluation prediction batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="AdamW learning rate")
    parser.add_argument("--max-length", type=int, default=192, help="Maximum tokens per example")
    parser.add_argument("--max-vocab-size", type=int, default=50000, help="Maximum vocabulary size")
    parser.add_argument("--embedding-dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden layer dimension")
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
    device = get_device(args.device)
    print(f"Using device: {describe_device(device)}")

    train_rows = load_labeled_split(args.train_dir)
    vocab = build_vocab(train_rows, max_vocab_size=args.max_vocab_size)

    print("Training expanded-form PyTorch classifier...")
    expanded_model = train_label_model(
        rows=train_rows,
        label_key="expanded",
        vocab=vocab,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
    )
    expanded_model_path = model_dir / "expanded_torch_model.pt"
    save_model_bundle(expanded_model, expanded_model_path, vocab)

    print("Training base-form PyTorch classifier...")
    base_model = train_label_model(
        rows=train_rows,
        label_key="base",
        vocab=vocab,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
    )
    base_model_path = model_dir / "base_torch_model.pt"
    save_model_bundle(base_model, base_model_path, vocab)

    dev_expected_path = Path(args.dev_dir) / "expected.tsv"
    if dev_expected_path.exists():
        dev_rows = load_labeled_split(args.dev_dir)
        majority_dictionary = build_majority_dictionary(train_rows)
        majority_predictions = predict_rows(dev_rows, majority_dictionary)
        majority_scores = _score_predictions(dev_rows, majority_predictions)

        ml_predictions = predict_with_ml_models(
            rows=dev_rows,
            expanded_model=expanded_model,
            base_model=base_model,
            vocab=vocab,
            device=device,
            batch_size=args.predict_batch_size,
        )
        ml_scores = _score_predictions(dev_rows, ml_predictions)

        _print_scores("Majority baseline", majority_scores)
        _print_scores("PyTorch classifier", ml_scores)
    else:
        print(f"No expected.tsv in dev dir: {args.dev_dir}. Skipping evaluation.")

    print(f"Saved expanded model to: {expanded_model_path}")
    print(f"Saved base model to: {base_model_path}")


if __name__ == "__main__":
    main()

