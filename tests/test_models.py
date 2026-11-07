"""Tests for dataset loading/validation, splits, the experiment registry and the promoted artifact."""
import json

import joblib
import numpy as np
import pandas as pd
import pytest

import config
from experiments.model_registry import CANDIDATES, build_pipeline, full_grid
from src import data as D


# ---------------------------------------------------------------- dataset loading & validation
def test_raw_file_parses_to_expected_schema():
    df = D.load_raw()
    assert list(df.columns) == ["sentence", "label"]
    assert len(df) == 4846
    assert set(df["label"].unique()) == {0, 1, 2}


def test_validation_report_matches_known_facts():
    rep = D.validate(D.load_raw())
    assert rep["n_rows"] == 4846 and rep["missing_values"] == {"sentence": 0, "label": 0}
    assert rep["empty_or_whitespace_sentences"] == 0
    assert rep["class_counts"] == {"negative": 604, "neutral": 2879, "positive": 1363}
    assert rep["sentences_with_conflicting_labels"] == 2


def test_processed_file_exists_and_loads():
    df = D.load_processed()
    assert {"sentence", "label", "label_name"} <= set(df.columns) and len(df) == 4846


def test_validate_flags_problems_in_synthetic_data():
    df = pd.DataFrame({"sentence": ["a b", "a b", "  ", "c d"], "label": [0, 1, 1, 1]})
    rep = D.validate(df)
    assert rep["empty_or_whitespace_sentences"] == 1 and rep["sentences_with_conflicting_labels"] == 1
    assert rep["duplicate_sentences_extra_copies"] == 1


def test_load_raw_rejects_unknown_labels(tmp_path, monkeypatch):
    f = tmp_path / "raw.txt"
    f.write_text("Some sentence@excited\n", encoding="latin-1")
    monkeypatch.setattr(config, "RAW_FILE", f)
    with pytest.raises(ValueError):
        D.load_raw()


def test_clean_drops_conflicts_and_dedupes():
    df = pd.DataFrame({"sentence": ["x", "x", "y", "y", "z"], "label": [0, 1, 2, 2, 1]})
    out = D.clean(df)
    assert sorted(out["sentence"]) == ["y", "z"] and len(out) == 2


# ---------------------------------------------------------------- splits
def test_splits_are_disjoint_complete_and_stratified():
    df, tr, va, te = D.get_splits()
    assert len(df) == 4836
    assert len(set(tr) | set(va) | set(te)) == len(df) == len(tr) + len(va) + len(te)
    assert not (set(tr) & set(va) or set(tr) & set(te) or set(va) & set(te))
    shares = [df.loc[i, "label"].value_counts(normalize=True).sort_index().values for i in (tr, va, te)]
    assert np.allclose(shares[0], shares[1], atol=0.01) and np.allclose(shares[0], shares[2], atol=0.01)
    assert abs(len(tr) / len(df) - 0.70) < 0.01 and abs(len(va) / len(df) - 0.15) < 0.01


def test_splits_are_reproducible():
    a, b = D.get_splits(), D.get_splits()
    assert all(np.array_equal(x, y) for x, y in zip(a[1:], b[1:]))


# ---------------------------------------------------------------- registry
def test_registry_has_the_six_required_candidates():
    assert set(CANDIDATES) == {"DummyClassifier", "MultinomialNB", "ComplementNB", "LogisticRegression", "LinearSVC", "SGDClassifier"}


@pytest.mark.parametrize("name", list(CANDIDATES))
def test_every_candidate_builds_a_pipeline_that_fits_and_predicts(name):
    pipe = build_pipeline(name)
    assert [s for s, _ in pipe.steps] == ["tfidf", "clf"]
    for key in full_grid(name):
        assert key in pipe.get_params()
    X = ["profit rose sharply", "net loss widened", "meeting held in helsinki", "sales grew", "costs fell", "board appointed"] * 2
    pipe.fit(X, [2, 0, 1, 2, 2, 1] * 2)
    assert set(pipe.predict(X)) <= {0, 1, 2}


# ---------------------------------------------------------------- promoted artifact
def test_models_dir_holds_only_the_promoted_model():
    assert sorted(p.name for p in config.MODELS_DIR.iterdir()) == ["best_model.joblib", "metadata.json"]


def test_artifact_loads_and_predicts():
    model = joblib.load(config.MODEL_PATH)
    assert set(model.predict(["profit rose", "loss widened"])) <= {0, 1, 2}


def test_metadata_records_required_fields():
    meta = json.loads(config.METADATA_PATH.read_text(encoding="utf-8"))
    for key in ("model_name", "model_config", "training_date_utc", "dataset", "random_seed", "feature_extraction", "training_split",
                "validation_score", "final_test_metrics", "class_labels", "margin_bands"):
        assert key in meta, key
    assert meta["class_labels"] == ["negative", "neutral", "positive"] and meta["random_seed"] == config.RANDOM_SEED
    assert 0 < meta["final_test_metrics"]["macro_f1"] <= 1


def test_experiment_outputs_exist_and_agree_with_metadata():
    comp = pd.read_csv(config.RESULTS_DIR / "comparison.csv")
    metrics = json.loads((config.RESULTS_DIR / "metrics.json").read_text(encoding="utf-8"))
    meta = json.loads(config.METADATA_PATH.read_text(encoding="utf-8"))
    assert len(comp) == 6 and comp["selected"].sum() == 1
    assert comp.loc[comp["selected"], "model"].iloc[0] == metrics["winner"] == meta["model_name"]
    assert (config.CONFUSION_DIR / "DummyClassifier.png").exists()
    # selection used validation macro-F1: the winner has the max validation score
    assert comp["val_macro_f1"].round(4).max() == comp.loc[comp["selected"], "val_macro_f1"].round(4).iloc[0]
