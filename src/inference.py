"""Reusable sentiment prediction helpers."""

from dataclasses import dataclass

try:
    from .artifacts import load_tokenizer, load_trained_model
    from .config import MAX_LEN, PREDICTION_THRESHOLD
    from .preprocess import clean_text, text_to_padded_sequence
except ImportError:
    from artifacts import load_tokenizer, load_trained_model
    from config import MAX_LEN, PREDICTION_THRESHOLD
    from preprocess import clean_text, text_to_padded_sequence


@dataclass(frozen=True)
class PredictionResult:
    label: str
    positive_probability: float
    confidence: float


class SentimentPredictor:
    def __init__(self, model=None, tokenizer=None, max_len: int = MAX_LEN):
        self.model = model or load_trained_model()
        self.tokenizer = tokenizer or load_tokenizer()
        self.max_len = max_len

    def predict(self, review: str, threshold: float = PREDICTION_THRESHOLD) -> PredictionResult:
        cleaned_review = clean_text(review)
        padded_review = text_to_padded_sequence(
            cleaned_review,
            tokenizer=self.tokenizer,
            max_len=self.max_len,
        )
        positive_probability = float(self.model.predict(padded_review, verbose=0)[0][0])
        label = "Positive" if positive_probability >= threshold else "Negative"
        confidence = positive_probability if label == "Positive" else 1 - positive_probability
        return PredictionResult(
            label=label,
            positive_probability=positive_probability,
            confidence=confidence,
        )
