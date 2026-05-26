from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from src.preprocess import preprocess_row


def _read_tsv_lines(path: Path) -> List[List[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    rows: List[List[str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, raw in enumerate(f, start=1):
            line = raw.rstrip("\n\r")
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                raise ValueError(
                    f"Invalid TSV row in {path} at line {line_number}: expected >=2 columns"
                )
            rows.append(parts)
    return rows


def load_labeled_split(split_dir: str | Path) -> List[Dict[str, str]]:
    """Load paired `in.tsv` + `expected.tsv` from a split directory."""
    split_path = Path(split_dir)
    in_rows = _read_tsv_lines(split_path / "in.tsv")
    expected_rows = _read_tsv_lines(split_path / "expected.tsv")

    if len(in_rows) != len(expected_rows):
        raise ValueError(
            f"Row count mismatch in {split_path}: in.tsv={len(in_rows)} vs expected.tsv={len(expected_rows)}"
        )

    records: List[Dict[str, str]] = []
    for idx, (in_row, out_row) in enumerate(zip(in_rows, expected_rows), start=1):
        abbr = in_row[0].strip()
        context = in_row[1].strip()
        expanded = out_row[0].strip()
        base = out_row[1].strip()

        if not abbr or not context or not expanded or not base:
            continue

        records.append(
            {
                "abbr": abbr,
                "context": context,
                "expanded": expanded,
                "base": base,
                "input_text": preprocess_row(abbr, context),
                "row_id": str(idx),
            }
        )

    return records


def load_unlabeled_input(in_path: str | Path) -> List[Dict[str, str]]:
    """Load test-style `in.tsv` with abbreviation + context columns."""
    path = Path(in_path)
    rows = _read_tsv_lines(path)
    records: List[Dict[str, str]] = []

    for idx, row in enumerate(rows, start=1):
        abbr = row[0].strip()
        context = row[1].strip()
        if not abbr or not context:
            continue

        records.append(
            {
                "abbr": abbr,
                "context": context,
                "input_text": preprocess_row(abbr, context),
                "row_id": str(idx),
            }
        )

    return records


def write_predictions_tsv(predictions: List[Dict[str, str]], output_path: str | Path) -> None:
    """Write output in expected two-column format without header."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8", newline="") as f:
        for row in predictions:
            f.write(f"{row['expanded']}\t{row['base']}\n")

