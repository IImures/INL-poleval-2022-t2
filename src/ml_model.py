from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from src.baseline import predict_for_abbr


def _safe_print(message: str) -> None:
    """Print debug text without crashing on Windows legacy console encodings."""
    print(message.encode("ascii", errors="backslashreplace").decode("ascii"))


def build_tfidf_classifier(random_state: int = 42) -> Pipeline:
    """Build a memory-efficient TF-IDF + LinearSVC text classifier.

    The feature set combines:
    - word n-grams for context-level clues,
    - character n-grams for Polish inflection and abbreviation patterns.

    LinearSVC is used instead of LogisticRegression because the task has many
    possible output labels; multinomial LogisticRegression can require a very
    large dense optimization workspace for this dataset.
    """
    return Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        (
                            "word_tfidf",
                            TfidfVectorizer(
                                analyzer="word",
                                ngram_range=(1, 3),
                                min_df=1,
                                max_features=10000,
                            ),
                        ),
                        (
                            "char_tfidf",
                            TfidfVectorizer(
                                analyzer="char_wb",
                                ngram_range=(3, 6),
                                min_df=1,
                                max_features=20000,
                            ),
                        ),
                    ]
                ),
            ),
            (
                "clf",
                LinearSVC(
                    random_state=random_state,
                    max_iter=5000,
                    dual="auto",
                ),
            ),
        ]
    )


def train_label_model(rows: Iterable[Dict[str, str]], label_key: str) -> Pipeline:
    """Train a classifier for either `expanded` or `base` labels."""
    row_list = list(rows)
    x_train = [row["input_text"] for row in row_list]
    y_train = [row[label_key] for row in row_list]

    model = build_tfidf_classifier()
    model.fit(x_train, y_train)
    return model


def predict_with_ml_models(
    rows: Iterable[Dict[str, str]],
    expanded_model: Pipeline,
    base_model: Pipeline,
    majority_dictionary: Dict[str, Dict[str, str]],
    candidate_dictionary: Dict[str, Dict[str, List[str]]] | None = None,
) -> List[Dict[str, str]]:
    """Predict expanded/base forms with ML models and majority fallback."""
    row_list = list(rows)
    x = [row["input_text"] for row in row_list]

    expanded_predictions = expanded_model.predict(x)
    base_predictions = base_model.predict(x)

    predictions: List[Dict[str, str]] = []
    for row, expanded, base in zip(row_list, expanded_predictions, base_predictions):
        expanded = str(expanded).strip()
        base = str(base).strip()
        ml_expanded = expanded
        ml_base = base

        candidates = (candidate_dictionary or {}).get(row["abbr"])
        impossible_expanded = bool(candidates) and expanded not in candidates["expanded"]
        impossible_base = bool(candidates) and base not in candidates["base"]

        if not expanded or not base or impossible_expanded or impossible_base:
            fallback_expanded, fallback_base = predict_for_abbr(row["abbr"], majority_dictionary)
            _safe_print(
                "Dictionary fallback used | "
                f"row_id={row.get('row_id', '')} | "
                f"abbr={row['abbr']} | "
                f"ml=({ml_expanded!r}, {ml_base!r}) | "
                f"dictionary=({fallback_expanded!r}, {fallback_base!r}) | "
                f"reason="
                f"empty={not ml_expanded or not ml_base}, "
                f"impossible_expanded={impossible_expanded}, "
                f"impossible_base={impossible_base}"
            )
            expanded, base = fallback_expanded, fallback_base

        predictions.append(
            {
                "expanded": expanded,
                "base": base,
                "abbr": row["abbr"],
                "row_id": row.get("row_id", ""),
            }
        )

    return predictions


def split_prediction_columns(predictions: Iterable[Dict[str, str]]) -> Tuple[List[str], List[str]]:
    """Return expanded/base lists from prediction dictionaries."""
    prediction_list = list(predictions)
    return [row["expanded"] for row in prediction_list], [row["base"] for row in prediction_list]
