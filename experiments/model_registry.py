"""Central registry of candidate models.

To ADD a candidate: add one `Candidate(...)` entry to CANDIDATES.
To REMOVE one: delete its entry. Nothing else in the project needs to change.

Every candidate shares the same TF-IDF front end and is wrapped in a scikit-learn Pipeline,
so the vectoriser is fitted only on whatever data the pipeline is fitted on (no leakage).
"""
from dataclasses import dataclass, field
from typing import Callable

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

import config
from src.preprocessing import TOKEN_PATTERN, normalize_text

# Fixed TF-IDF settings (EDA Section I). ngram_range is tuned per candidate below.
TFIDF_FIXED = dict(preprocessor=normalize_text, lowercase=False, token_pattern=TOKEN_PATTERN,
                   min_df=2, sublinear_tf=True)

# Grid entries shared by every real model: unigrams vs unigrams+bigrams.
SHARED_GRID = {"tfidf__ngram_range": [(1, 1), (1, 2)]}

SEED = config.RANDOM_SEED


@dataclass
class Candidate:
    name: str
    description: str
    make_classifier: Callable[[], object]
    param_grid: dict = field(default_factory=dict)   # extra grid on top of SHARED_GRID


CANDIDATES = {c.name: c for c in [
    Candidate("DummyClassifier", "Majority-class baseline (always predicts the most frequent class)",
              lambda: DummyClassifier(strategy="most_frequent"), param_grid={}),
    Candidate("MultinomialNB", "Multinomial Naive Bayes on TF-IDF",
              lambda: MultinomialNB(), {"clf__alpha": [0.1, 0.3, 1.0]}),
    Candidate("ComplementNB", "Complement Naive Bayes (designed for imbalanced text)",
              lambda: ComplementNB(), {"clf__alpha": [0.1, 0.3, 1.0]}),
    Candidate("LogisticRegression", "Multinomial logistic regression (L2)",
              lambda: LogisticRegression(max_iter=2000, random_state=SEED),
              {"clf__C": [1, 10, 100], "clf__class_weight": [None, "balanced"]}),
    Candidate("LinearSVC", "Linear support vector classifier (one-vs-rest, L2)",
              lambda: LinearSVC(max_iter=10000, random_state=SEED),
              {"clf__C": [0.1, 1, 10], "clf__class_weight": [None, "balanced"]}),
    Candidate("SGDClassifier", "Linear model trained by SGD with log-loss",
              lambda: SGDClassifier(loss="log_loss", max_iter=1000, tol=1e-3, random_state=SEED),
              {"clf__alpha": [1e-5, 1e-4, 1e-3], "clf__class_weight": [None, "balanced"]}),
]}


def build_pipeline(name: str) -> Pipeline:
    """Fresh, unfitted pipeline: TF-IDF -> classifier."""
    return Pipeline([("tfidf", TfidfVectorizer(**TFIDF_FIXED)),
                     ("clf", CANDIDATES[name].make_classifier())])


def full_grid(name: str) -> dict:
    """Grid for a candidate. The dummy baseline ignores text, so it gets no grid."""
    if name == "DummyClassifier":
        return {}
    return {**SHARED_GRID, **CANDIDATES[name].param_grid}
