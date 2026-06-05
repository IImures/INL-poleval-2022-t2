import re


def normalize_text(text: str) -> str:
    """Normalize whitespace and lowercase while preserving punctuation and <mask>."""
    text = (text or "").strip().lower()
    text = re.sub(r"\s*<mask>\s*", " <mask> ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def extract_mask_window(context: str, window_size: int = 8) -> tuple[str, str, str]:
    """Extract left/right/local-window text around the <mask> token."""
    context_norm = normalize_text(context)
    tokens = context_norm.split()

    if "<mask>" not in tokens:
        return "", "", ""

    mask_index = tokens.index("<mask>")
    left_tokens = tokens[max(0, mask_index - window_size) : mask_index]
    right_tokens = tokens[mask_index + 1 : mask_index + 1 + window_size]
    window_tokens = left_tokens + ["<mask>"] + right_tokens

    return " ".join(left_tokens), " ".join(right_tokens), " ".join(window_tokens)


def preprocess_row(abbr: str, context: str, window_size: int = 8) -> str:
    """Build model input text in a shared train/predict format."""
    abbr_norm = normalize_text(abbr)
    context_norm = normalize_text(context)
    left, right, window = extract_mask_window(context_norm, window_size=window_size)
    return (
        f"ABBR={abbr_norm} "
        f"LEFT={left} "
        f"RIGHT={right} "
        f"WINDOW={window} "
        f"CONTEXT={context_norm}"
    )

