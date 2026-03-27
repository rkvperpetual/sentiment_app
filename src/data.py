"""Dataset loading and preparation helpers."""

import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from .config import RANDOM_STATE, TEST_SIZE
    from .paths import DATASET_PATH
    from .preprocess import clean_text
except ImportError:
    from config import RANDOM_STATE, TEST_SIZE
    from paths import DATASET_PATH
    from preprocess import clean_text


def load_dataset(path=DATASET_PATH) -> pd.DataFrame:
    dataset = pd.read_csv(path)
    dataset["review"] = dataset["review"].apply(clean_text)
    dataset["sentiment"] = dataset["sentiment"].map({"positive": 1, "negative": 0})
    return dataset


def split_dataset(dataset: pd.DataFrame, test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE):
    return train_test_split(
        dataset["review"],
        dataset["sentiment"],
        test_size=test_size,
        random_state=random_state,
        stratify=dataset["sentiment"],
    )
