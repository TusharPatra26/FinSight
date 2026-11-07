"""Metrics, timing, confusion matrices and comparison outputs for the experiments."""
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_recall_fscore_support)

import config

LABEL_IDS = list(range(len(config.LABEL_NAMES)))   # consistent order everywhere: negative, neutral, positive
PAPER, CHAR, COPPER, FOREST, GOLD, BURG, GRID = "#F2EFE6", "#252724", "#D97745", "#34483B", "#D6B66C", "#914D4B", "#D9D4C8"


def compute_metrics(y_true, y_pred) -> dict:
    """All required metrics for one set of predictions."""
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=LABEL_IDS, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(p.mean()), "macro_recall": float(r.mean()),
        "macro_f1": float(f.mean()),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", labels=LABEL_IDS, zero_division=0)),
        "per_class": {n: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
                      for i, n in enumerate(config.LABEL_NAMES)},
    }


def macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="macro", labels=LABEL_IDS, zero_division=0))


def report_text(y_true, y_pred) -> str:
    return classification_report(y_true, y_pred, labels=LABEL_IDS, target_names=config.LABEL_NAMES, zero_division=0, digits=3)


def conf_matrix(y_true, y_pred) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=LABEL_IDS)


def time_fit(make_pipeline, X, y, repeats=3) -> tuple:
    """Fit a fresh pipeline `repeats` times; return (fitted_pipeline, median_seconds)."""
    times, model = [], None
    for _ in range(repeats):
        model = make_pipeline()
        t0 = time.perf_counter(); model.fit(X, y); times.append(time.perf_counter() - t0)
    return model, float(np.median(times))


def time_predict(model, X, repeats=5) -> tuple:
    """Return (predictions, median_seconds_for_whole_set, ms_per_sentence)."""
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter(); pred = model.predict(X); times.append(time.perf_counter() - t0)
    med = float(np.median(times))
    return pred, med, 1000 * med / len(X)


def model_complexity(pipe) -> dict:
    """Number of TF-IDF features and learned classifier parameters (where defined)."""
    clf = pipe.named_steps["clf"]
    n_feat = len(pipe.named_steps["tfidf"].vocabulary_)
    if hasattr(clf, "coef_"): n_par = int(clf.coef_.size + np.size(getattr(clf, "intercept_", [])))
    elif hasattr(clf, "feature_log_prob_"): n_par = int(clf.feature_log_prob_.size + clf.class_log_prior_.size)
    else: n_par = 0
    return {"n_features": int(n_feat), "n_parameters": n_par}


def plot_confusion(cm, title, path) -> None:
    """Counts annotated with row-percentages (recall per true class)."""
    pct = cm / cm.sum(axis=1, keepdims=True).clip(min=1) * 100
    annot = np.array([[f"{cm[i, j]}\n{pct[i, j]:.0f}%" for j in range(cm.shape[1])] for i in range(cm.shape[0])])
    fig, ax = plt.subplots(figsize=(4.6, 4))
    sns.heatmap(pct, annot=annot, fmt="", cmap=sns.light_palette(FOREST, as_cmap=True), vmin=0, vmax=100, cbar=False, ax=ax,
                xticklabels=config.LABEL_NAMES, yticklabels=config.LABEL_NAMES, linewidths=1, linecolor=PAPER)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title, fontsize=10, fontweight="bold"); ax.grid(False)
    fig.patch.set_facecolor(PAPER); fig.tight_layout(); fig.savefig(path, dpi=130, facecolor=PAPER); plt.close(fig)


def plot_comparison(df: pd.DataFrame, path) -> None:
    """Macro-F1, accuracy, weighted-F1 (validation vs test) plus training and inference time."""
    d = df.sort_values("val_macro_f1")
    fig, ax = plt.subplots(2, 3, figsize=(14, 7)); ax = ax.ravel()
    y = np.arange(len(d)); h = 0.38
    for a, (title, vcol, tcol) in zip(ax[:3], [("Macro F1", "val_macro_f1", "test_macro_f1"),
                                               ("Accuracy", "val_accuracy", "test_accuracy"),
                                               ("Weighted F1", "val_weighted_f1", "test_weighted_f1")]):
        a.barh(y + h / 2, d[vcol], h, color=COPPER, label="validation"); a.barh(y - h / 2, d[tcol], h, color=FOREST, label="test")
        a.set_yticks(y); a.set_yticklabels(d.model if a is ax[0] else [""] * len(d)); a.set_xlim(0, 1); a.set_title(title); a.grid(axis="y", visible=False)
    ax[0].legend(loc="lower right", fontsize=8)
    ax[3].barh(y, d.train_time_s, color=GOLD); ax[3].set_title("Training time (seconds, train split)")
    ax[4].barh(y, d.inference_ms_per_sentence, color=BURG); ax[4].set_title("Inference (ms per sentence)")
    for a in ax[3:5]: a.set_yticks(y); a.set_yticklabels(d.model if a is ax[3] else [""] * len(d)); a.grid(axis="y", visible=False)
    ax[5].axis("off")
    ax[5].text(0, .9, "Selection: highest validation macro-F1\n(tie-break: CV macro-F1, then\nfaster inference, then name).\n\nTest bars are reported AFTER selection\nand did not influence it.\n\nDummy = majority-class baseline.", va="top", fontsize=10)
    fig.patch.set_facecolor(PAPER)
    for a in ax: a.set_facecolor(PAPER)
    fig.tight_layout(); fig.savefig(path, dpi=130, facecolor=PAPER); plt.close(fig)
