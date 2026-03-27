"""Train and save the sentiment analysis model."""

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

try:
    from .artifacts import save_tokenizer
    from .config import (
        BATCH_SIZE,
        EARLY_STOPPING_PATIENCE,
        EPOCHS,
        LR_REDUCTION_FACTOR,
        LR_REDUCTION_PATIENCE,
        MIN_LEARNING_RATE,
        VALIDATION_SPLIT,
    )
    from .data import load_dataset, split_dataset
    from .model import build_model
    from .paths import MODEL_PATH, TOKENIZER_PATH
    from .preprocess import create_tokenizer, texts_to_padded_sequences
except ImportError:
    from artifacts import save_tokenizer
    from config import (
        BATCH_SIZE,
        EARLY_STOPPING_PATIENCE,
        EPOCHS,
        LR_REDUCTION_FACTOR,
        LR_REDUCTION_PATIENCE,
        MIN_LEARNING_RATE,
        VALIDATION_SPLIT,
    )
    from data import load_dataset, split_dataset
    from model import build_model
    from paths import MODEL_PATH, TOKENIZER_PATH
    from preprocess import create_tokenizer, texts_to_padded_sequences


def build_callbacks():
    return [
        EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=LR_REDUCTION_FACTOR,
            patience=LR_REDUCTION_PATIENCE,
            min_lr=MIN_LEARNING_RATE,
        ),
        ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]


def train_model():
    dataset = load_dataset()
    x_train, x_test, y_train, y_test = split_dataset(dataset)

    tokenizer = create_tokenizer()
    tokenizer.fit_on_texts(x_train)

    x_train_padded = texts_to_padded_sequences(x_train, tokenizer=tokenizer)
    x_test_padded = texts_to_padded_sequences(x_test, tokenizer=tokenizer)

    model = build_model()
    history = model.fit(
        x_train_padded,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        callbacks=build_callbacks(),
    )

    loss, accuracy = model.evaluate(x_test_padded, y_test, verbose=0)
    model.save(MODEL_PATH)
    save_tokenizer(tokenizer, TOKENIZER_PATH)

    return {
        "model": model,
        "tokenizer": tokenizer,
        "history": history,
        "loss": float(loss),
        "accuracy": float(accuracy),
    }


def main():
    result = train_model()
    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved tokenizer to: {TOKENIZER_PATH}")
    print(f"Test Accuracy: {result['accuracy'] * 100:.2f}%")


if __name__ == "__main__":
    main()
