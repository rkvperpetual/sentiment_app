"""Text cleaning and sequence preparation utilities."""

import html
import re

try:
    from .config import MAX_LEN, MAX_WORDS, OOV_TOKEN
except ImportError:
    from config import MAX_LEN, MAX_WORDS, OOV_TOKEN

HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
UNWANTED_CHARACTER_PATTERN = re.compile(r"[^a-zA-Z0-9\s']")
WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_text(text: str) -> str:
    cleaned_text = html.unescape(str(text))
    cleaned_text = HTML_TAG_PATTERN.sub(" ", cleaned_text)
    cleaned_text = UNWANTED_CHARACTER_PATTERN.sub(" ", cleaned_text)
    cleaned_text = WHITESPACE_PATTERN.sub(" ", cleaned_text)
    return cleaned_text.lower().strip()


def create_tokenizer(num_words: int = MAX_WORDS):
    from tensorflow.keras.preprocessing.text import Tokenizer

    return Tokenizer(num_words=num_words, oov_token=OOV_TOKEN)


def texts_to_padded_sequences(texts, tokenizer, max_len: int = MAX_LEN):
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    sequences = tokenizer.texts_to_sequences(texts)
    return pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")


def text_to_padded_sequence(text: str, tokenizer, max_len: int = MAX_LEN):
    return texts_to_padded_sequences([text], tokenizer=tokenizer, max_len=max_len)
