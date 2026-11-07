"""Shared text preprocessing. Used by training (experiments) AND inference, so they can never drift apart."""
import re

# Keeps single-character tokens, numbers and the % / currency symbols. The EDA showed that
# number/percent/currency presence is informative, and sklearn's default pattern drops them.
TOKEN_PATTERN = r"(?u)\b\w+\b|[%$€£]"

_WS = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Collapse whitespace and lower-case. Deliberately does NOT remove stopwords,
    numbers or punctuation-bearing tokens (see analysis/eda_report.md, Section I)."""
    return _WS.sub(" ", str(text)).strip().lower()
