"""Dataset download, loading and validation for Financial PhraseBank (sentences_50agree).

Run `python -m src.data` to download, validate and write the processed file.
"""
import json
import sys
import zipfile

import numpy as np
import pandas as pd

import config


def download_raw(force: bool = False):
    """Download the original authors' archive from the Hugging Face Hub and
    save ONLY the 50%-agreement text file, byte-for-byte, to data/raw/.

    Why not load_dataset(...)? Current `datasets` versions refuse script-based
    datasets, and takala/financial_phrasebank is one. The same Hub repo hosts
    the authors' original zip, which is exactly what the script used to read.
    """
    if config.RAW_FILE.exists() and not force:
        return config.RAW_FILE
    from huggingface_hub import hf_hub_download  # imported lazily: only needed to download

    config.DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    archive = hf_hub_download(config.DATASET_REPO, config.DATASET_ARCHIVE, repo_type="dataset")
    with zipfile.ZipFile(archive) as zf:
        # Read one named member into memory; never extract the whole archive.
        config.RAW_FILE.write_bytes(zf.read(config.DATASET_MEMBER))
    return config.RAW_FILE


def load_raw() -> pd.DataFrame:
    """Parse the raw file into a DataFrame with `sentence` and `label` (int id).

    Raw format: one record per line, `sentence@label`, latin-1 encoded. A
    sentence may itself contain '@', so we split on the LAST '@'.
    """
    rows = []
    with open(config.RAW_FILE, encoding="latin-1") as fh:
        for line in fh:
            line = line.rstrip("\r\n")
            if not line:
                continue
            sentence, _, label_name = line.rpartition("@")
            rows.append((sentence, label_name))
    df = pd.DataFrame(rows, columns=["sentence", "label_name"])
    unknown = set(df["label_name"]) - set(config.LABEL_NAMES)
    if unknown:
        raise ValueError(f"Unexpected labels in raw file: {unknown}")
    df["label"] = df["label_name"].map(config.LABEL_TO_ID).astype("int64")
    return df[["sentence", "label"]]


def validate(df: pd.DataFrame) -> dict:
    """Return a data-quality summary. Nothing is removed here; EDA decides that."""
    stripped = df["sentence"].str.strip()
    dup_mask = df.duplicated(subset="sentence", keep=False)
    labels_per_sentence = df.groupby("sentence")["label"].nunique()
    conflicting = labels_per_sentence[labels_per_sentence > 1]
    counts = df["label"].value_counts().sort_index()
    return {
        "dataset": f"{config.DATASET_REPO}/{config.DATASET_CONFIG}",
        "n_rows": int(len(df)),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing_values": {c: int(v) for c, v in df.isna().sum().items()},
        "empty_or_whitespace_sentences": int((stripped == "").sum()),
        "sentences_with_leading_trailing_whitespace": int((df["sentence"] != stripped).sum()),
        "exact_duplicate_rows": int(df.duplicated().sum()),
        "rows_in_duplicate_sentence_groups": int(dup_mask.sum()),
        "duplicate_sentences_extra_copies": int(df.duplicated(subset="sentence").sum()),
        "sentences_with_conflicting_labels": int(len(conflicting)),
        "class_counts": {config.LABEL_NAMES[i]: int(n) for i, n in counts.items()},
        "class_percentages": {config.LABEL_NAMES[i]: round(100 * n / len(df), 2) for i, n in counts.items()},
    }


def load_processed() -> pd.DataFrame:
    """Load the processed CSV, building it first if it does not exist."""
    if not config.PROCESSED_FILE.exists():
        prepare()
    return pd.read_csv(config.PROCESSED_FILE)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Modelling-ready copy of the data (decisions documented in analysis/eda_report.md).

    1. Sentences that appear with CONFLICTING labels are dropped entirely: the
       annotation is genuinely ambiguous, and keeping either label is a guess.
    2. Remaining exact duplicate sentences (same label) keep one copy. Otherwise a
       sentence could land in both train and test and inflate scores (leakage).
    """
    n_labels = df.groupby("sentence")["label"].transform("nunique")
    df = df[n_labels == 1]
    return df.drop_duplicates(subset="sentence").reset_index(drop=True)


def get_splits():
    """Reproducible stratified 70/15/15 split. Returns (clean_df, train_idx, val_idx, test_idx).

    Indices refer to rows of the cleaned DataFrame. Same seed => same indices, so every
    model and the notebook see identical splits.
    """
    from sklearn.model_selection import train_test_split

    df = clean(load_processed())
    idx = df.index.values
    train_val, test = train_test_split(
        idx, test_size=0.15, stratify=df["label"], random_state=config.RANDOM_SEED)
    train, val = train_test_split(
        train_val, test_size=0.15 / 0.85, stratify=df.loc[train_val, "label"],
        random_state=config.RANDOM_SEED)
    return df, np.sort(train), np.sort(val), np.sort(test)


def prepare(force: bool = False) -> dict:
    """Download (if needed), validate, and write processed data + report.

    Processing here is deliberately minimal: whitespace trimming only. No rows are
    dropped; duplicates/conflicts are investigated in the EDA before any decision.
    """
    download_raw(force=force)
    df = load_raw()
    report = validate(df)
    df["sentence"] = df["sentence"].str.strip()
    df["label_name"] = df["label"].map(dict(enumerate(config.LABEL_NAMES)))
    config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.PROCESSED_FILE, index=False)
    config.VALIDATION_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(prepare(force="--force" in sys.argv), indent=2))
