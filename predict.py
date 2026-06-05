from __future__ import annotations

import argparse
from pathlib import Path

from src.data_io import load_unlabeled_input, write_predictions_tsv
from src.ml_model import (
    describe_device,
    get_device,
    load_model_bundle,
    predict_with_ml_models,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict abbreviation expansions using saved majority baseline."
    )
    parser.add_argument("--input", required=True, help="Path to test/input in.tsv file")
    parser.add_argument("--model-dir", default="models", help="Directory with trained model artifacts")
    parser.add_argument("--output", required=True, help="Path to output predictions TSV")
    parser.add_argument("--device", default="auto", help="Torch device: auto, cuda, or cpu")
    parser.add_argument("--batch-size", type=int, default=256, help="Prediction batch size")
    parser.add_argument(
        "--method",
        choices=["torch", "ml"],
        default="torch",
        help="Prediction method. `ml` is accepted as an alias for `torch`.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.method == "ml":
        args.method = "torch"

    model_dir = Path(args.model_dir)
    device = get_device(args.device)

    input_rows = load_unlabeled_input(args.input)

    expanded_model_path = model_dir / "expanded_torch_model.pt"
    base_model_path = model_dir / "base_torch_model.pt"

    required_paths = [expanded_model_path, base_model_path]
    missing_paths = [str(path) for path in required_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(
            "Missing PyTorch model artifacts. Run train_model.py first. "
            f"Missing: {missing_paths}"
        )

    expanded_model = load_model_bundle(expanded_model_path, device)
    base_model = load_model_bundle(base_model_path, device)
    vocab = expanded_model["vocab"]
    assert isinstance(vocab, dict)
    predictions = predict_with_ml_models(
        rows=input_rows,
        expanded_model=expanded_model,
        base_model=base_model,
        vocab=vocab,
        device=device,
        batch_size=args.batch_size,
    )

    write_predictions_tsv(predictions, args.output)

    print(f"Loaded rows: {len(input_rows)}")
    print(f"Prediction method: {args.method}")
    print(f"Device: {describe_device(device)}")
    print(f"Saved predictions to: {args.output}")


if __name__ == "__main__":
    main()

