"""Central configuration for FinSight. Every path and constant lives here."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# --- Folders -----------------------------------------------------------------
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
ANALYSIS_DIR = BASE_DIR / "analysis"
EXPERIMENTS_DIR = BASE_DIR / "experiments"
RESULTS_DIR = EXPERIMENTS_DIR / "results"
CONFUSION_DIR = RESULTS_DIR / "confusion_matrices"
MODELS_DIR = BASE_DIR / "models"
UPLOADS_DIR = BASE_DIR / "uploads"
DATABASE_PATH = BASE_DIR / "database" / "finsight.db"

# --- Dataset -----------------------------------------------------------------
DATASET_REPO = "takala/financial_phrasebank"
DATASET_CONFIG = "sentences_50agree"          # never mixed with other agreement levels
DATASET_ARCHIVE = "data/FinancialPhraseBank-v1.0.zip"
DATASET_MEMBER = "FinancialPhraseBank-v1.0/Sentences_50Agree.txt"
RAW_FILE = DATA_RAW_DIR / "Sentences_50Agree.txt"          # untouched original bytes
PROCESSED_FILE = DATA_PROCESSED_DIR / "phrasebank_50agree.csv"
VALIDATION_REPORT = DATA_PROCESSED_DIR / "validation_report.json"

# Label ids follow the Hugging Face dataset card: 0=negative, 1=neutral, 2=positive.
LABEL_NAMES = ["negative", "neutral", "positive"]
LABEL_TO_ID = {name: i for i, name in enumerate(LABEL_NAMES)}

# --- Reproducibility ---------------------------------------------------------
RANDOM_SEED = 42
