"""Tests for src/inference.py and src/preprocessing.py."""
import pytest

from src.inference import (InputValidationError, ModelNotFoundError, SentimentModel, get_model, margin_from_scores,
                           validate_text)
from src.preprocessing import normalize_text

POSITIVE = "Operating profit rose to EUR 12.5 mn from EUR 8.1 mn in the previous year."
NEGATIVE = "The company reported a net loss and sales decreased by 15 % compared with last year."
NEUTRAL = "The company will hold its annual general meeting on 5 April in Helsinki."


@pytest.fixture(scope="module")
def model():
    return get_model()


# ---------------------------------------------------------------- preprocessing / validation
def test_normalize_text_lowercases_and_collapses_whitespace():
    assert normalize_text("  Profit\tROSE \n by 5 % ") == "profit rose by 5 %"


@pytest.mark.parametrize("bad", [None, 123, "", "   ", "!!!", "x" * 2001])
def test_validate_text_rejects_bad_input(bad):
    with pytest.raises(InputValidationError):
        validate_text(bad)


def test_validate_text_cleans_whitespace_and_control_chars():
    assert validate_text("  Profit \x00 rose\n today ") == "Profit rose today"


def test_margin_from_scores_is_gap_between_top_two():
    import numpy as np
    assert margin_from_scores(np.array([[0.1, 0.9, 0.4]]))[0] == pytest.approx(0.5)


# ---------------------------------------------------------------- loading
def test_model_loads_and_has_expected_metadata(model):
    assert model.labels == ["negative", "neutral", "positive"]
    import re
    assert model.model_name == "LinearSVC" and re.fullmatch(r"LinearSVC-[0-9a-f]{8}", model.model_id), model.model_id
    assert model.bands and len(model.bands["cutpoints"]) == 2


def test_missing_model_raises_helpful_error(tmp_path):
    with pytest.raises(ModelNotFoundError, match="run_experiments"):
        SentimentModel(tmp_path / "nope.joblib", tmp_path / "nope.json")


def test_corrupt_model_raises_model_error(tmp_path):
    bad = tmp_path / "bad.joblib"
    bad.write_bytes(b"not a model")
    with pytest.raises(ModelNotFoundError):
        SentimentModel(bad, tmp_path / "nope.json")


# ---------------------------------------------------------------- single prediction
def test_single_prediction_output_format(model):
    r = model.predict(POSITIVE)
    assert set(r) >= {"text", "sentiment", "score_type", "score_type_label", "score", "scores", "margin", "margin_band",
                      "explanation", "warnings", "model_id"}
    assert r["sentiment"] in model.labels
    assert set(r["scores"]) == set(model.labels)
    assert r["score"] == r["scores"][r["sentiment"]] == max(r["scores"].values())
    assert r["margin_band"] in ("Low", "Medium", "High")


def test_scores_are_not_labelled_as_probabilities(model):
    r = model.predict(POSITIVE)
    assert r["score_type"] == "decision_score"
    assert "not a probability" in r["score_type_label"]


@pytest.mark.parametrize("text,expected", [(POSITIVE, "positive"), (NEGATIVE, "negative"), (NEUTRAL, "neutral")])
def test_clear_examples_get_expected_class(model, text, expected):
    assert model.predict(text)["sentiment"] == expected


def test_explanation_structure_and_content(model):
    e = model.predict(NEGATIVE, top_k=5)["explanation"]
    assert e["available"] and e["predicted_class"] == "negative" and "association" in e["note"]
    assert all(t["weight"] > 0 for t in e["supporting"]) and all(t["weight"] < 0 for t in e["opposing"])
    assert len(e["supporting"]) <= 5
    assert {"loss", "decreased"} & {t["term"] for t in e["supporting"]}


def test_unknown_vocabulary_warns_and_is_low_confidence(model):
    r = model.predict("zzzz qqqq")
    assert r["warnings"] and r["margin_band"] == "Low" and r["explanation"]["supporting"] == []


def test_single_prediction_error_message_has_no_item_prefix(model):
    with pytest.raises(InputValidationError) as exc:
        model.predict("")
    assert "Item" not in str(exc.value)


# ---------------------------------------------------------------- batch prediction
def test_batch_matches_single_and_pipeline(model):
    texts = [POSITIVE, NEGATIVE, NEUTRAL]
    batch = model.predict_batch(texts)
    assert [b["sentiment"] for b in batch] == [model.predict(t)["sentiment"] for t in texts]
    expected = [model.labels[i] for i in model.pipeline.predict(texts)]
    assert [b["sentiment"] for b in batch] == expected


def test_batch_rejects_empty_list_and_bad_item(model):
    with pytest.raises(InputValidationError):
        model.predict_batch([])
    with pytest.raises(InputValidationError, match="Item 2"):
        model.predict_batch([POSITIVE, ""])


def test_inference_is_deterministic(model):
    assert model.predict(POSITIVE) == model.predict(POSITIVE)
