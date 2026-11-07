"""Centralised experiment runner. ONE script trains, tunes, compares, selects and promotes.

Run from the project root:   python -m experiments.run_experiments

Order of operations (this order is what protects the test set):
  Phase 1  For every candidate and every grid point: 5-fold CV on TRAIN, plus fit-on-train / score-on-VALIDATION.
  Phase 2  Per candidate, keep the grid point with best validation macro-F1. Pick the overall winner the same way.
           The test set has NOT been touched yet.
  Phase 3  Score every candidate's chosen model on TEST (reporting only), save comparison outputs.
  Phase 4  Refit the winner on TRAIN+VALIDATION, evaluate once on TEST, promote to models/.
"""
import json
import platform
import sys
import time
from datetime import datetime, timezone

import joblib
import pandas as pd
import sklearn
from sklearn.model_selection import ParameterGrid, StratifiedKFold, cross_val_score

import config
from experiments import evaluation as ev
from experiments.model_registry import CANDIDATES, TFIDF_FIXED, build_pipeline, full_grid
from src import data as D
from src.inference import margin_from_scores


def _json_params(params: dict) -> dict:
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in params.items()}


def _rank_key(r):  # higher val macro-F1, then higher CV macro-F1 (rounded to 4 dp to avoid float noise)
    return (-round(r["val_macro_f1"], 4), -round(r["cv_macro_f1_mean"], 4))


def _margin_bands(name, params, X, y, cv) -> dict:
    """Split the winner's confidence margin (top score minus runner-up) into Low/Medium/High terciles using
    5-fold out-of-fold scores, and measure the real accuracy inside each band. Used by src/inference.py."""
    import numpy as np
    from sklearn.model_selection import cross_val_predict
    pipe = build_pipeline(name).set_params(**params)
    method = "predict_proba" if hasattr(pipe.named_steps["clf"], "predict_proba") else "decision_function"
    S = cross_val_predict(pipe, X, y, cv=cv, method=method)
    margins = margin_from_scores(S)
    pred = S.argmax(axis=1)
    lo, hi = (float(q) for q in np.quantile(margins, [1 / 3, 2 / 3]))
    band = np.where(margins < lo, 0, np.where(margins < hi, 1, 2))
    yv = np.asarray(y)
    names = ("Low", "Medium", "High")
    return {"signal": f"top-two gap of {method}", "cutpoints": [round(lo, 4), round(hi, 4)],
            "method": "5-fold out-of-fold scores on train+validation; bands are terciles of the margin",
            "n_by_band": {n: int((band == i).sum()) for i, n in enumerate(names)},
            "accuracy_by_band": {n: round(float((pred[band == i] == yv[band == i]).mean()), 4) for i, n in enumerate(names)},
            "note": "Margin bands are an empirical reliability indicator, not calibrated probabilities."}


def main() -> None:
    t_start = time.perf_counter()
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True); config.CONFUSION_DIR.mkdir(parents=True, exist_ok=True)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for old in config.CONFUSION_DIR.glob("*.png"):   # only our own generated figures
        old.unlink()

    df, tr, va, te = D.get_splits()
    X, y = df["sentence"], df["label"]
    Xtr, ytr, Xva, yva, Xte, yte = X[tr], y[tr], X[va], y[va], X[te], y[te]
    print(f"Rows {len(df)} | train {len(tr)} | validation {len(va)} | test {len(te)} | seed {config.RANDOM_SEED}")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)

    # ---------------- Phase 1 + 2: tune on train/validation ----------------
    results, fitted = {}, {}
    for name, cand in CANDIDATES.items():
        grid = list(ParameterGrid(full_grid(name))) or [{}]
        search = []
        for params in grid:
            pipe = build_pipeline(name).set_params(**params)
            cvs = cross_val_score(pipe, Xtr, ytr, cv=cv, scoring="f1_macro")
            pipe.fit(Xtr, ytr); pv = pipe.predict(Xva)
            m = ev.compute_metrics(yva, pv)
            search.append({"params": params, "cv_macro_f1_mean": float(cvs.mean()), "cv_macro_f1_std": float(cvs.std()),
                           "val_macro_f1": m["macro_f1"], "val_accuracy": m["accuracy"]})
        best = sorted(search, key=_rank_key)[0]
        # Refit best config on train only, with timing; inference timed on VALIDATION (test still untouched).
        model, train_s = ev.time_fit(lambda p=best["params"], n=name: build_pipeline(n).set_params(**p), Xtr, ytr)
        pv, inf_s, inf_ms = ev.time_predict(model, Xva)
        results[name] = {"description": cand.description, "n_grid_points": len(grid), "best_params": _json_params(best["params"]),
                         "cv_macro_f1_mean": best["cv_macro_f1_mean"], "cv_macro_f1_std": best["cv_macro_f1_std"],
                         "validation": ev.compute_metrics(yva, pv), "train_time_s": train_s,
                         "inference_time_s_val_set": inf_s, "inference_ms_per_sentence": inf_ms,
                         **ev.model_complexity(model),
                         "search": [{**s, "params": _json_params(s["params"])} for s in search]}
        fitted[name] = model
        print(f"  {name:20s} grid={len(grid):2d}  CV macroF1={best['cv_macro_f1_mean']:.4f}±{best['cv_macro_f1_std']:.4f}  "
              f"VAL macroF1={results[name]['validation']['macro_f1']:.4f}  best={_json_params(best['params'])}")

    # ---------------- Phase 2: select the winner (no test data used) ----------------
    ranking = sorted(results, key=lambda n: (-round(results[n]["validation"]["macro_f1"], 4), -round(results[n]["cv_macro_f1_mean"], 4),
                                             results[n]["inference_ms_per_sentence"], n))
    winner = ranking[0]
    print(f"\nWINNER (validation macro-F1, tie-break CV macro-F1 -> faster inference -> name): {winner}")

    # ---------------- Phase 3: test-set reporting for all candidates ----------------
    for name, model in fitted.items():
        pt = model.predict(Xte); m = ev.compute_metrics(yte, pt)
        results[name]["test"] = m; results[name]["classification_report_test"] = ev.report_text(yte, pt)
        ev.plot_confusion(ev.conf_matrix(yte, pt), f"{name} (trained on train only) - test set", config.CONFUSION_DIR / f"{name}.png")
        results[name]["confusion_matrix_test"] = ev.conf_matrix(yte, pt).tolist()

    rows = []
    for name, r in results.items():
        t, v = r["test"], r["validation"]
        row = {"model": name, "selected": name == winner, "rank_by_validation": ranking.index(name) + 1,
               "best_params": json.dumps(r["best_params"]),
               "cv_macro_f1_mean": r["cv_macro_f1_mean"], "cv_macro_f1_std": r["cv_macro_f1_std"],
               "val_macro_f1": v["macro_f1"], "val_accuracy": v["accuracy"], "val_weighted_f1": v["weighted_f1"],
               "test_accuracy": t["accuracy"], "test_macro_precision": t["macro_precision"], "test_macro_recall": t["macro_recall"],
               "test_macro_f1": t["macro_f1"], "test_weighted_f1": t["weighted_f1"]}
        for cls in config.LABEL_NAMES:
            for k in ("precision", "recall", "f1"):
                row[f"test_{cls}_{k}"] = t["per_class"][cls][k]
        row.update(train_time_s=r["train_time_s"], inference_time_s_val_set=r["inference_time_s_val_set"],
                   inference_ms_per_sentence=r["inference_ms_per_sentence"], n_features=r["n_features"], n_parameters=r["n_parameters"])
        rows.append(row)
    comp = pd.DataFrame(rows).sort_values("rank_by_validation")
    comp.to_csv(config.RESULTS_DIR / "comparison.csv", index=False)
    ev.plot_comparison(comp, config.RESULTS_DIR / "comparison.png")

    # ---------------- Phase 4: refit winner on train+validation, final test, promote ----------------
    trval = sorted(list(tr) + list(va))
    best_params = {k: (tuple(v) if isinstance(v, list) else v) for k, v in results[winner]["best_params"].items()}
    final, final_train_s = ev.time_fit(lambda: build_pipeline(winner).set_params(**best_params), X[trval], y[trval])
    pf, f_inf_s, f_inf_ms = ev.time_predict(final, Xte)
    final_metrics = ev.compute_metrics(yte, pf)
    ev.plot_confusion(ev.conf_matrix(yte, pf), f"PRODUCTION: {winner} (train+validation) - test set", config.CONFUSION_DIR / f"production_{winner}.png")
    print(f"PRODUCTION model test: accuracy={final_metrics['accuracy']:.4f} macroF1={final_metrics['macro_f1']:.4f} weightedF1={final_metrics['weighted_f1']:.4f}")

    # Margin bands for the app's confidence indicator: out-of-fold scores on train+validation ONLY (test untouched).
    margin_bands = _margin_bands(winner, best_params, X[trval], y[trval], cv)
    print("Margin bands (out-of-fold, train+val):", margin_bands["cutpoints"], margin_bands["accuracy_by_band"])

    tf = final.named_steps["tfidf"]
    metadata = {
        "model_name": winner,
        "model_config": {"classifier": winner, "classifier_params": final.named_steps["clf"].get_params(), "tuned_params": results[winner]["best_params"]},
        "training_date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {"source": config.DATASET_REPO, "configuration": config.DATASET_CONFIG, "rows_raw": 4846, "rows_after_cleaning": len(df),
                    "cleaning": "src.data.clean(): drop conflicting-label sentences, de-duplicate identical sentences"},
        "random_seed": config.RANDOM_SEED,
        "feature_extraction": {"vectorizer": "TfidfVectorizer", "ngram_range": list(tf.ngram_range), "min_df": tf.min_df,
                               "sublinear_tf": tf.sublinear_tf, "token_pattern": tf.token_pattern, "lowercase_in_preprocessor": True,
                               "preprocessor": "src.preprocessing.normalize_text", "stopword_removal": False, "vocabulary_size": len(tf.vocabulary_)},
        "training_split": {"method": "stratified 70/15/15 via src.data.get_splits()", "n_train": len(tr), "n_validation": len(va), "n_test": len(te),
                           "production_model_fitted_on": f"train+validation ({len(trval)} rows); test set held out"},
        "selection": {"metric": "validation macro-F1", "tie_breaks": ["CV macro-F1 (train)", "lower inference time", "name"]},
        "validation_score": {"macro_f1": results[winner]["validation"]["macro_f1"], "accuracy": results[winner]["validation"]["accuracy"],
                             "note": "model fitted on train only, scored on validation", "cv_macro_f1_mean": results[winner]["cv_macro_f1_mean"]},
        "final_test_metrics": final_metrics,
        "margin_bands": margin_bands,
        "class_labels": config.LABEL_NAMES, "label_ids": ev.LABEL_IDS,
        "environment": {"python": platform.python_version(), "scikit_learn": sklearn.__version__},
    }
    joblib.dump(final, config.MODELS_DIR / "best_model.joblib", compress=3)
    (config.MODELS_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    all_metrics = {
        "meta": {"seed": config.RANDOM_SEED, "n_train": len(tr), "n_validation": len(va), "n_test": len(te),
                 "cv": "StratifiedKFold(5, shuffle, seed) on train", "selection_metric": "validation macro-F1",
                 "tie_breaks": ["CV macro-F1", "inference ms/sentence", "name"],
                 "notes": "Per-candidate test metrics come from models fitted on TRAIN only and are reported after selection. "
                          "Inference time is measured on the validation set. 'production' is the winner refit on train+validation.",
                 "tfidf_fixed": {k: str(v) for k, v in TFIDF_FIXED.items()}},
        "ranking_by_validation": ranking, "winner": winner,
        "models": results,
        "production": {"model": winner, "fitted_on_rows": len(trval), "test_metrics": final_metrics, "train_time_s": final_train_s,
                       "inference_ms_per_sentence_test": f_inf_ms, "classification_report_test": ev.report_text(yte, pf),
                       "confusion_matrix_test": ev.conf_matrix(yte, pf).tolist()},
    }
    (config.RESULTS_DIR / "metrics.json").write_text(json.dumps(all_metrics, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved comparison.csv, metrics.json, comparison.png, confusion matrices, models/best_model.joblib, models/metadata.json "
          f"({time.perf_counter() - t_start:.0f}s total)")


if __name__ == "__main__":
    sys.exit(main())
