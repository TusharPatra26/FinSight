"""Reusable inference: load the PROMOTED model and predict. Never trains anything.

Scores are labelled honestly. The production model is a LinearSVC, which has no
probabilities, so we report its raw *decision scores* (higher = more confident for that
class; NOT probabilities) plus a Low/Medium/High *margin band*. The band cut-points and
their measured accuracy come from cross-validation on train+validation and are stored in
models/metadata.json (see experiments/run_experiments.py). If the promoted model ever has
`predict_proba`, probabilities are reported instead and labelled as such.
"""
import json
import threading

import joblib
import numpy as np

import config

MARGIN_LABELS = ("Low", "Medium", "High")
SCORE_TYPE_LABELS = {
    "decision_score": "Model decision score (SVM margin units, not a probability)",
    "probability": "Model probability estimate (uncalibrated)",
}
EXPLANATION_NOTE = (
    "Influential words come from the linear model: each word's TF-IDF weight multiplied by the model's "
    "coefficient for the predicted class. This shows statistical association learned from training data, "
    "not causation or meaning. It ignores word order beyond two-word phrases, and scores are for the predicted "
    "class only (not a head-to-head comparison with the runner-up class)."
)


class InferenceError(Exception):
    """Base class for inference problems."""


class ModelNotFoundError(InferenceError):
    """The promoted model artifact is missing or unreadable."""


class InputValidationError(InferenceError, ValueError):
    """The supplied text is not acceptable input."""


def validate_text(text, max_chars: int = config.MAX_TEXT_CHARS) -> str:
    """Return cleaned text or raise InputValidationError."""
    if text is None:
        raise InputValidationError("No text was provided.")
    if not isinstance(text, str):
        raise InputValidationError("Text must be a string.")
    printable = "".join(ch for ch in text if ch.isprintable() or ch.isspace())
    cleaned = " ".join(printable.split())
    if not cleaned:
        raise InputValidationError("The text is empty. Please enter a financial news sentence.")
    if not any(ch.isalnum() for ch in cleaned):
        raise InputValidationError("The text must contain letters or numbers.")
    if len(cleaned) > max_chars:
        raise InputValidationError(f"The text is too long ({len(cleaned)} characters; the limit is {max_chars}).")
    return cleaned


def margin_from_scores(scores: np.ndarray) -> np.ndarray:
    """Gap between the top two class scores for each row (a simple, model-agnostic confidence signal)."""
    s = np.sort(np.asarray(scores, dtype=float), axis=1)
    return s[:, -1] - s[:, -2]


def score_matrix(pipeline, X):
    """Return (score_type, scores) for transformed features X, preferring probabilities when they exist."""
    clf = pipeline.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        return "probability", clf.predict_proba(X)
    if hasattr(clf, "decision_function"):
        return "decision_score", clf.decision_function(X)
    raise InferenceError(f"{type(clf).__name__} provides neither probabilities nor decision scores.")


class SentimentModel:
    """Wrapper around the promoted pipeline (TF-IDF -> classifier) and its metadata."""

    def __init__(self, model_path=None, metadata_path=None):
        model_path = model_path or config.MODEL_PATH
        metadata_path = metadata_path or config.METADATA_PATH
        try:
            self.pipeline = joblib.load(model_path)
            self.metadata = json.loads(open(metadata_path, encoding="utf-8").read())
        except FileNotFoundError as exc:
            raise ModelNotFoundError(
                f"Model artifact not found ({exc.filename}). Create it with: python -m experiments.run_experiments") from exc
        except Exception as exc:  # corrupt pickle, bad json, missing sklearn class, ...
            raise ModelNotFoundError(f"The model artifact could not be loaded: {exc}") from exc
        steps = getattr(self.pipeline, "named_steps", {})
        if "tfidf" not in steps or "clf" not in steps:
            raise ModelNotFoundError("The model artifact is not a TF-IDF -> classifier pipeline.")
        self.labels = list(self.metadata["class_labels"])
        self._classes = list(steps["clf"].classes_)
        self._features = steps["tfidf"].get_feature_names_out()
        self.model_name = self.metadata["model_name"]
        self.model_id = f"{self.model_name}@{self.metadata['training_date_utc'][:10]}"
        self.bands = self.metadata.get("margin_bands")

    # ------------------------------------------------------------------ public API
    def predict(self, text, top_k: int = 5) -> dict:
        """Predict one text. Raises InputValidationError for unacceptable input."""
        return self.predict_batch([validate_text(text)], top_k=top_k)[0]

    def predict_batch(self, texts, top_k: int = 3) -> list:
        """Predict many texts at once (vectorised). Any invalid item raises InputValidationError."""
        texts = list(texts)
        if not texts:
            raise InputValidationError("No texts were provided.")
        cleaned = []
        for i, t in enumerate(texts):
            try:
                cleaned.append(validate_text(t))
            except InputValidationError as exc:
                raise InputValidationError(f"Item {i + 1}: {exc}") from exc
        steps = self.pipeline.named_steps
        X = steps["tfidf"].transform(cleaned)
        pred_ids = steps["clf"].predict(X)
        score_type, S = score_matrix(self.pipeline, X)
        margins = margin_from_scores(S)
        out = []
        for i, text in enumerate(cleaned):
            col = self._classes.index(pred_ids[i])
            explanation = self._explain(X[i], col, top_k)
            warnings = []
            band = self._band(margins[i])
            if X[i].nnz == 0:
                warnings.append("None of the words in this text are in the model's vocabulary; the prediction is a default guess.")
                band = "Low" if band else None   # never show confidence for a text the model cannot read
            out.append({
                "text": text,
                "sentiment": self.labels[int(pred_ids[i])],
                "score_type": score_type,
                "score_type_label": SCORE_TYPE_LABELS[score_type],
                "score": round(float(S[i, col]), 4),
                "scores": {self.labels[int(c)]: round(float(S[i, j]), 4) for j, c in enumerate(self._classes)},
                "margin": round(float(margins[i]), 4),
                "margin_band": band,
                "explanation": explanation,
                "warnings": warnings,
                "model_id": self.model_id,
            })
        return out

    # ------------------------------------------------------------------ helpers
    def _band(self, margin):
        if not self.bands:
            return None
        lo, hi = self.bands["cutpoints"]
        return MARGIN_LABELS[0 if margin < lo else 1 if margin < hi else 2]

    def _explain(self, x_row, class_col: int, top_k: int) -> dict:
        clf = self.pipeline.named_steps["clf"]
        base = {"predicted_class": self.labels[int(self._classes[class_col])], "note": EXPLANATION_NOTE}
        if not hasattr(clf, "coef_"):
            return {**base, "available": False, "supporting": [], "opposing": []}
        row = x_row.tocoo()
        contrib = row.data * clf.coef_[class_col][row.col]
        order = np.argsort(contrib)
        terms = [{"term": str(self._features[row.col[j]]), "weight": round(float(contrib[j]), 4)} for j in order]
        supporting = [t for t in reversed(terms) if t["weight"] > 0][:top_k]
        opposing = [t for t in terms if t["weight"] < 0][:top_k]
        return {**base, "available": True, "supporting": supporting, "opposing": opposing}


_lock = threading.Lock()
_cached = None


def get_model() -> SentimentModel:
    """Load the promoted model once per process."""
    global _cached
    with _lock:
        if _cached is None:
            _cached = SentimentModel()
        return _cached


def reset_cache() -> None:
    global _cached
    with _lock:
        _cached = None
