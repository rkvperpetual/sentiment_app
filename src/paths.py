"""Filesystem paths used across the project."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

DATASET_PATH = DATA_DIR / "IMDB_Dataset.csv"
MODEL_PATH = PROJECT_ROOT / "lstm_sentiment.h5"
TOKENIZER_PATH = PROJECT_ROOT / "tokenizer.pkl"
