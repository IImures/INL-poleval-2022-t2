from __future__ import annotations

from typing import Dict, Iterable, List


def _normalize_for_eval(value: str) -> str:
    return (value or "").strip().lower()


def accuracy_case_insensitive(y_true: Iterable[str], y_pred: Iterable[str]) -> float:
    true_list: List[str] = [_normalize_for_eval(v) for v in y_true]
    pred_list: List[str] = [_normalize_for_eval(v) for v in y_pred]

    if not true_list:
        return 0.0

    correct = sum(1 for t, p in zip(true_list, pred_list) if t == p)
    return correct / len(true_list)


def calculate_scores(
    true_expanded: Iterable[str],
    pred_expanded: Iterable[str],
    true_base: Iterable[str],
    pred_base: Iterable[str],
) -> Dict[str, float]:
    af = accuracy_case_insensitive(true_expanded, pred_expanded)
    ab = accuracy_case_insensitive(true_base, pred_base)
    final = 0.25 * af + 0.75 * ab

    return {"Af": af, "Ab": ab, "Final": final}

