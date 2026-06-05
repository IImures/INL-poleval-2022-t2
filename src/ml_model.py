from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline


def build_tfidf_classifier(random_state: int = 42) -> Pipeline:
    """Build a TF-IDF + LogisticRegression text classifier.

    The feature set combines:
    - word n-grams for context-level clues,
    - character n-grams for Polish inflection and abbreviation patterns.

    The `saga` solver is used because it works better with sparse TF-IDF
    matrices than the default solver for this many-class text task.
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
                LogisticRegression(
                    solver="saga",
                    max_iter=1000,
                    random_state=random_state,
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
) -> List[Dict[str, str]]:
    """Predict expanded/base forms with ML models only, without dictionary fallback."""
    row_list = list(rows)
    x = [row["input_text"] for row in row_list]

    expanded_predictions = expanded_model.predict(x)
    base_predictions = base_model.predict(x)

    predictions: List[Dict[str, str]] = []
    for row, expanded, base in zip(row_list, expanded_predictions, base_predictions):
        expanded = str(expanded).strip()
        base = str(base).strip()

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
