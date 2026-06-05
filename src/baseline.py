from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Tuple


def build_majority_dictionary(rows: Iterable[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    """Build majority expanded/base mapping for each abbreviation."""
    expanded_counts: Dict[str, Counter] = defaultdict(Counter)
    base_counts: Dict[str, Counter] = defaultdict(Counter)

    for row in rows:
        abbr = row["abbr"]
        expanded_counts[abbr][row["expanded"]] += 1
        base_counts[abbr][row["base"]] += 1

    majority: Dict[str, Dict[str, str]] = {}
    for abbr in expanded_counts:
        majority_expanded = expanded_counts[abbr].most_common(1)[0][0]
        majority_base = base_counts[abbr].most_common(1)[0][0]
        majority[abbr] = {"expanded": majority_expanded, "base": majority_base}

    return majority


def predict_for_abbr(
    abbr: str,
    majority_dictionary: Dict[str, Dict[str, str]],
) -> Tuple[str, str]:
    """Predict expanded/base for abbreviation using majority dictionary fallback."""
    if abbr in majority_dictionary:
        entry = majority_dictionary[abbr]
        return entry["expanded"], entry["base"]

    # Unseen abbreviation fallback: return abbreviation itself for both forms.
    return abbr, abbr


def predict_rows(
    rows: Iterable[Dict[str, str]],
    majority_dictionary: Dict[str, Dict[str, str]],
) -> List[Dict[str, str]]:
    predictions: List[Dict[str, str]] = []

    for row in rows:
        expanded, base = predict_for_abbr(row["abbr"], majority_dictionary)
        predictions.append(
            {
                "expanded": expanded,
                "base": base,
                "abbr": row["abbr"],
                "row_id": row.get("row_id", ""),
            }
        )

    return predictions

