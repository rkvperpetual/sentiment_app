"""Model definition for sentiment classification."""

from tensorflow import keras

try:
    from .config import (
        DENSE_DROPOUT_RATE,
        DENSE_UNITS,
        EMBEDDING_DIM,
        LSTM_UNITS,
        MAX_LEN,
        MAX_WORDS,
        SPATIAL_DROPOUT_RATE,
    )
except ImportError:
    from config import (
        DENSE_DROPOUT_RATE,
        DENSE_UNITS,
        EMBEDDING_DIM,
        LSTM_UNITS,
        MAX_LEN,
        MAX_WORDS,
        SPATIAL_DROPOUT_RATE,
    )


def build_model(max_words: int = MAX_WORDS, max_len: int = MAX_LEN):
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(max_len,)),
            keras.layers.Embedding(input_dim=max_words, output_dim=EMBEDDING_DIM),
            keras.layers.SpatialDropout1D(SPATIAL_DROPOUT_RATE),
            keras.layers.Bidirectional(keras.layers.LSTM(LSTM_UNITS, return_sequences=True)),
            keras.layers.GlobalMaxPool1D(),
            keras.layers.Dense(DENSE_UNITS, activation="relu"),
            keras.layers.Dropout(DENSE_DROPOUT_RATE),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )
    return model
