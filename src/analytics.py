"""Reusable aggregation and NLP analytics (CSV validation, distributions, financial vocabulary, experiment results).

Everything the Flask routes need beyond prediction lives here, so routes stay thin.
"""
import json
import re
from collections import Counter

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

import config


class CSVValidationError(ValueError):
    """The uploaded CSV cannot be analysed; the message is safe to show to the user."""


TEXT_COLUMN_CANDIDATES = ("text", "sentence", "headline", "title", "news", "content", "description", "body")

# Small, transparent, manually defined lexicon (identical to analysis/eda.ipynb, Section F).
# NOT a complete financial dictionary; exact word-form matching; some words are ambiguous (share, interest, price).
LEXICON = {
    "Revenue & profitability": "revenue revenues sales profit profits profitability profitable earnings income margin margins ebitda ebit turnover".split(),
    "Losses & expenses": "loss losses costs cost expenses expense deficit writedown impairment charges layoffs layoff redundancies".split(),
    "Growth & expansion": "growth grow grew growing increase increased increases rose rise rising expansion expand expanded expanding boost improved improvement higher".split(),
    "Debt & liabilities": "debt debts loan loans borrowing liabilities liability credit bond bonds leverage convertible interest".split(),
    "Investments & acquisitions": "investment investments invest invested acquisition acquisitions acquire acquired acquires merger stake buyout divest divestment takeover".split(),
    "Market performance": "shares stock stocks market price prices index trading volume".split(),
    "Risk & uncertainty": "risk risks uncertainty uncertain volatility volatile weak weaker weakness warning warned concern concerns pressure downturn recession crisis decline declined decrease decreased fall fell drop dropped lower".split(),
    "Dividends & shareholder returns": "dividend dividends payout buyback repurchase shareholders shareholder yield distribution".split(),
}
_TERM_TO_CATEGORY = {w: c for c, ws in LEXICON.items() for w in ws}
_WORD = re.compile(r"[a-z]+")


# --------------------------------------------------------------------------- CSV input
def read_csv_texts(source, text_column=None, max_rows=config.MAX_BATCH_ROWS, max_chars=config.MAX_TEXT_CHARS):
    """Read and validate a CSV of texts. Returns (texts, report). Raises CSVValidationError with a friendly message.

    `source` is a path or file-like object. Empty cells and over-long texts are skipped and counted.
    Files with more than `max_rows` data rows are rejected (never silently truncated).
    """
    try:
        df = pd.read_csv(source, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    except UnicodeDecodeError:
        try:
            if hasattr(source, "seek"):
                source.seek(0)
            df = pd.read_csv(source, dtype=str, keep_default_na=False, encoding="latin-1")
        except Exception as exc:
            raise CSVValidationError("The file is not readable text (unknown encoding).") from exc
    except pd.errors.EmptyDataError as exc:
        raise CSVValidationError("The file is empty.") from exc
    except pd.errors.ParserError as exc:
        raise CSVValidationError("The file could not be parsed as CSV. Check that it is a comma-separated file with a header row.") from exc
    except Exception as exc:
        raise CSVValidationError("The file could not be read as CSV.") from exc

    if df.shape[1] == 0 or len(df) == 0:
        raise CSVValidationError("The CSV has no data rows. It needs a header row followed by at least one text row.")
    if len(df) > max_rows:
        raise CSVValidationError(f"The CSV has {len(df):,} rows; the limit is {max_rows:,}. Split it into smaller files.")

    cols = {str(c).strip().lower(): c for c in df.columns}
    if text_column:
        key = text_column.strip().lower()
        if key not in cols:
            raise CSVValidationError(f"Column '{text_column}' was not found. Columns in the file: {', '.join(map(str, df.columns))}.")
        column = cols[key]
    else:
        column = next((cols[c] for c in TEXT_COLUMN_CANDIDATES if c in cols), None)
        if column is None and df.shape[1] == 1:
            column = df.columns[0]
        if column is None:
            raise CSVValidationError("Could not tell which column holds the text. Name it 'text' or 'sentence', or choose the column. "
                                     f"Columns in the file: {', '.join(map(str, df.columns))}.")

    texts, empty, too_long = [], 0, 0
    for raw in df[column]:
        t = " ".join(str(raw).split())
        if not t or not any(ch.isalnum() for ch in t):
            empty += 1
        elif len(t) > max_chars:
            too_long += 1
        else:
            texts.append(t)
    if not texts:
        raise CSVValidationError("No usable text was found in the selected column (all cells were empty or too long).")
    return texts, {"text_column": str(column), "rows_in_file": int(len(df)), "rows_analysed": len(texts),
                   "skipped_empty": empty, "skipped_too_long": too_long}


# --------------------------------------------------------------------------- aggregation
def sentiment_distribution(labels) -> dict:
    """Counts and percentages for each class, in the fixed class order. Accepts labels or a {label: count} dict."""
    c = labels if isinstance(labels, dict) else Counter(labels)
    total = sum(c.get(n, 0) for n in config.LABEL_NAMES)
    return {n: {"count": c.get(n, 0), "percent": round(100 * c.get(n, 0) / total, 1) if total else 0.0} for n in config.LABEL_NAMES}


def financial_term_counts(texts, top_n: int = 10) -> dict:
    """How many texts mention each lexicon category, and the most frequent lexicon terms."""
    n = len(texts)
    cat_sentences, term_counts = Counter(), Counter()
    for t in texts:
        words = set(_WORD.findall(t.lower()))
        hit = {w for w in words if w in _TERM_TO_CATEGORY}
        for w in hit:
            term_counts[w] += 1
        for c in {_TERM_TO_CATEGORY[w] for w in hit}:
            cat_sentences[c] += 1
    cats = [{"category": c, "texts": cat_sentences.get(c, 0), "percent": round(100 * cat_sentences.get(c, 0) / n, 1) if n else 0.0} for c in LEXICON]
    return {"categories": cats, "top_terms": [{"term": w, "count": k, "category": _TERM_TO_CATEGORY[w]} for w, k in term_counts.most_common(top_n)]}


def top_words_by_sentiment(texts, labels, top_n: int = 8) -> dict:
    """Most frequent non-stopword words in texts of each predicted class (descriptive counts only)."""
    counters = {n: Counter() for n in config.LABEL_NAMES}
    for t, lab in zip(texts, labels):
        if lab in counters:
            counters[lab].update(w for w in _WORD.findall(t.lower()) if len(w) > 2 and w not in ENGLISH_STOP_WORDS)
    return {n: [{"word": w, "count": k} for w, k in counters[n].most_common(top_n)] for n in config.LABEL_NAMES}


def batch_summary(results) -> dict:
    """Aggregate view of a batch of prediction results (as returned by SentimentModel.predict_batch)."""
    texts = [r["text"] for r in results]
    labels = [r["sentiment"] for r in results]
    bands = Counter(r["margin_band"] for r in results if r["margin_band"])
    return {"n": len(results), "distribution": sentiment_distribution(labels),
            "margin_bands": {b: bands.get(b, 0) for b in ("Low", "Medium", "High")},
            "financial_vocabulary": financial_term_counts(texts),
            "top_words": top_words_by_sentiment(texts, labels)}


# --------------------------------------------------------------------------- dataset / experiment results
def dataset_overview() -> dict:
    """Facts about the cleaned training corpus (from data/processed)."""
    from src import data as D
    df = D.clean(D.load_processed())
    words = df["sentence"].str.split().str.len()
    dist = sentiment_distribution(df["label_name"])
    return {"rows": int(len(df)), "distribution": dist, "mean_words": round(float(words.mean()), 1), "median_words": float(words.median())}


def load_metadata() -> dict:
    return json.loads(config.METADATA_PATH.read_text(encoding="utf-8"))


def load_experiment_results() -> dict:
    """Experiment outputs for the Model Laboratory. Returns {} fields as None if experiments have not been run."""
    metrics_path, comp_path = config.RESULTS_DIR / "metrics.json", config.RESULTS_DIR / "comparison.csv"
    if not (metrics_path.exists() and comp_path.exists()):
        return {"available": False}
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    comparison = pd.read_csv(comp_path).to_dict(orient="records")
    return {"available": True, "metrics": metrics, "comparison": comparison}
