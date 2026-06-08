import re


def normalize_text(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def preprocess_row(abbr: str, context: str) -> str:
    abbr_norm = normalize_text(abbr)
    context_norm = normalize_text(context)
    return f"ABBR={abbr_norm} CONTEXT={context_norm}"

