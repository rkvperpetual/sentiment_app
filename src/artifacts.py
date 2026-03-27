"""Helpers for loading and saving trained artifacts."""

from pathlib import Path
import pickle

from tensorflow.keras.models import load_model

try:
    from .paths import MODEL_PATH, TOKENIZER_PATH
except ImportError:
    from paths import MODEL_PATH, TOKENIZER_PATH


def save_tokenizer(tokenizer, path: Path = TOKENIZER_PATH) -> None:
    with path.open("wb") as file:
        pickle.dump(tokenizer, file)


def load_tokenizer(path: Path = TOKENIZER_PATH):
    with path.open("rb") as file:
        return pickle.load(file)


def load_trained_model(path: Path = MODEL_PATH):
    return load_model(path, compile=False)
