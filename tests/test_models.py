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


# ---------------------------------------------------------------- evaluation metrics (hand-computed expectations)
def test_compute_metrics_matches_hand_calculation():
    from experiments import evaluation as ev
    m = ev.compute_metrics([0, 1, 2, 1, 1], [0, 1, 2, 1, 0])
    assert m["accuracy"] == pytest.approx(0.8)
    # class 0: P=1/2, R=1/1; class 1: P=2/2, R=2/3; class 2: P=R=1
    assert m["per_class"]["negative"]["precision"] == pytest.approx(0.5) and m["per_class"]["negative"]["recall"] == pytest.approx(1.0)
    assert m["per_class"]["neutral"]["precision"] == pytest.approx(1.0) and m["per_class"]["neutral"]["recall"] == pytest.approx(2 / 3)
    assert m["macro_precision"] == pytest.approx((0.5 + 1 + 1) / 3) and m["macro_recall"] == pytest.approx((1 + 2 / 3 + 1) / 3)
    assert m["macro_f1"] == pytest.approx((2 / 3 + 0.8 + 1) / 3)
    assert m["weighted_f1"] == pytest.approx((2 / 3 * 1 + 0.8 * 3 + 1 * 1) / 5)
    assert [m["per_class"][c]["support"] for c in ("negative", "neutral", "positive")] == [1, 3, 1]


def test_majority_baseline_has_known_macro_f1():
    from experiments import evaluation as ev
    y = [1] * 6 + [0] * 2 + [2] * 2          # 60% neutral
    m = ev.compute_metrics(y, [1] * 10)
    assert m["accuracy"] == pytest.approx(0.6) and m["macro_f1"] == pytest.approx((0 + 0.75 + 0) / 3)


def test_confusion_matrix_uses_fixed_label_order_even_when_a_class_is_absent():
    from experiments import evaluation as ev
    cm = ev.conf_matrix([0, 0, 1], [0, 1, 1])
    assert cm.shape == (3, 3) and cm[0].tolist() == [1, 1, 0] and cm[1].tolist() == [0, 1, 0] and cm[2].sum() == 0


def test_model_complexity_counts_features_and_parameters():
    from experiments import evaluation as ev
    pipe = build_pipeline("LinearSVC").fit(["profit rose", "loss widened", "meeting held"] * 4, [2, 0, 1] * 4)
    c = ev.model_complexity(pipe)
    clf = pipe.named_steps["clf"]
    assert c["n_features"] == len(pipe.named_steps["tfidf"].vocabulary_)
    assert c["n_parameters"] == clf.coef_.size + clf.intercept_.size
    dummy = build_pipeline("DummyClassifier").fit(["profit rose", "loss widened"] * 3, [2, 0] * 3)
    assert ev.model_complexity(dummy)["n_parameters"] == 0


def test_timing_helpers_return_predictions_and_positive_times():
    from experiments import evaluation as ev
    X, y = ["profit rose", "loss widened", "meeting held"] * 4, [2, 0, 1] * 4
    model, secs = ev.time_fit(lambda: build_pipeline("MultinomialNB"), X, y, repeats=2)
    pred, total, per_sentence = ev.time_predict(model, X, repeats=2)
    assert secs > 0 and total > 0 and len(pred) == len(X) and per_sentence == pytest.approx(1000 * total / len(X))


def test_plots_are_written_to_the_requested_path(tmp_path):
    from experiments import evaluation as ev
    ev.plot_confusion(np.array([[3, 1, 0], [0, 5, 1], [1, 0, 4]]), "demo", tmp_path / "cm.png")
    comp = pd.DataFrame({"model": ["A", "B"], "val_macro_f1": [.7, .6], "test_macro_f1": [.65, .6], "val_accuracy": [.8, .7], "test_accuracy": [.78, .7],
                         "val_weighted_f1": [.79, .7], "test_weighted_f1": [.77, .69], "train_time_s": [.1, .2], "inference_ms_per_sentence": [.03, .04]})
    ev.plot_comparison(comp, tmp_path / "cmp.png")
    assert (tmp_path / "cm.png").stat().st_size > 1000 and (tmp_path / "cmp.png").stat().st_size > 1000


# ---------------------------------------------------------------- experiment runner helpers
def test_selection_rank_key_orders_by_validation_then_cv():
    from experiments.run_experiments import _rank_key
    rows = [{"val_macro_f1": .70, "cv_macro_f1_mean": .90}, {"val_macro_f1": .75, "cv_macro_f1_mean": .50}, {"val_macro_f1": .75, "cv_macro_f1_mean": .60}]
    best = sorted(rows, key=_rank_key)[0]
    assert best == rows[2]                              # highest validation, then highest CV


def test_margin_bands_are_ordered_and_partition_the_data():
    from sklearn.model_selection import StratifiedKFold
    from experiments.run_experiments import _margin_bands
    pos = ["profit rose sharply", "sales increased strongly", "earnings grew well", "strong growth reported"]
    neg = ["net loss widened", "sales decreased sharply", "costs rose and profit fell", "weak demand hurt results"]
    neu = ["meeting held in helsinki", "company is headquartered in espoo", "board appointed new director", "shares listed on exchange"]
    X = [f"{t} {i}" for i in range(8) for t in pos + neg + neu]
    y = [2] * 4 + [0] * 4 + [1] * 4
    y = y * 8
    b = _margin_bands("LinearSVC", {"tfidf__ngram_range": (1, 2)}, pd.Series(X), pd.Series(y), StratifiedKFold(5, shuffle=True, random_state=1))
    lo, hi = b["cutpoints"]
    assert lo <= hi and set(b["accuracy_by_band"]) == {"Low", "Medium", "High"} and sum(b["n_by_band"].values()) == len(X)
    assert all(0 <= v <= 1 for v in b["accuracy_by_band"].values())


def test_prepare_builds_processed_files_without_downloading(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(config, "PROCESSED_FILE", tmp_path / "p.csv")
    monkeypatch.setattr(config, "VALIDATION_REPORT", tmp_path / "v.json")
    report = D.prepare()                                # raw file already exists, so nothing is downloaded
    assert report["n_rows"] == 4846 and len(pd.read_csv(tmp_path / "p.csv")) == 4846
    assert json.loads((tmp_path / "v.json").read_text(encoding="utf-8"))["class_counts"]["neutral"] == 2879
    assert not (pd.read_csv(tmp_path / "p.csv")["sentence"] != pd.read_csv(tmp_path / "p.csv")["sentence"].str.strip()).any()


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
