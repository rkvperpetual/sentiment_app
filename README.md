---
title: IMDB Sentiment Analysis
emoji: 🎬
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8501
---

# Sentiment Analysis on IMDB Movie Reviews

This project is a deep learning based sentiment analysis application that classifies movie reviews as positive or negative. It uses the IMDB movie review dataset, trains a Bidirectional LSTM model with TensorFlow/Keras, and provides a simple Streamlit web app for interactive predictions.

## Features

- Binary sentiment classification for movie reviews
- Text preprocessing and tokenization pipeline
- Bidirectional LSTM model built with TensorFlow/Keras
- Trained model and tokenizer saved for reuse
- Streamlit interface for live review analysis
- Shared config, path, training, and inference modules for cleaner maintenance
- Basic unit tests for preprocessing behavior

## Project Structure

```text
Sentiment Analysis/
|-- data/
|   |-- IMDB_Dataset.csv
|-- src/
|   |-- __init__.py
|   |-- app.py
|   |-- artifacts.py
|   |-- config.py
|   |-- data.py
|   |-- inference.py
|   |-- model.py
|   |-- paths.py
|   |-- preprocess.py
|   `-- train.py
|-- tests/
|   `-- test_preprocess.py
|-- .dockerignore
|-- lstm_sentiment.h5
|-- tokenizer.pkl
|-- .gitignore
|-- Dockerfile
|-- requirements.txt
`-- README.md
```

## Dataset

The project uses the `IMDB_Dataset.csv` file stored in the `data/` directory.

- Total reviews: 50,000
- Classes: `positive`, `negative`
- Balanced dataset: 25,000 positive and 25,000 negative reviews

## How It Works

### 1. Preprocessing

The preprocessing step:

- removes HTML tags
- removes non-letter characters
- converts text to lowercase
- tokenizes reviews using a fixed vocabulary
- pads all sequences to the same length

Current preprocessing settings:

- `MAX_WORDS = 20000`
- `MAX_LEN = 300`
- tokenizer uses an out-of-vocabulary token for unseen words

### 2. Model Architecture

The model is defined in `src/model.py` and contains:

- an Embedding layer
- a SpatialDropout1D layer for regularization
- a Bidirectional LSTM layer with 128 units
- a GlobalMaxPool1D layer
- a Dense layer with ReLU activation
- a Dropout layer
- a final Dense layer with Sigmoid activation

This makes it a binary classifier that outputs the probability that a review is positive.

### 3. Training

Training is handled in `src/train.py`:

- loads and cleans the IMDB dataset
- converts sentiment labels to numeric values
- splits the data into train and test sets
- creates and fits the tokenizer on training text
- trains the neural network
- uses callbacks for early stopping, learning-rate reduction, and best-model checkpointing
- evaluates it on the test set
- saves:
  - `lstm_sentiment.h5`
  - `tokenizer.pkl`

### 4. Inference App

The Streamlit app in `src/app.py`:

- loads a shared predictor built from the saved model and tokenizer
- accepts a review from the user
- preprocesses the input
- predicts sentiment
- displays the result and probability score

## Code Organization

- `src/config.py`: shared settings such as sequence length, vocabulary size, and training defaults
- `src/paths.py`: central place for dataset and artifact paths
- `src/data.py`: dataset loading and train/test splitting helpers
- `src/preprocess.py`: text cleaning, tokenizer creation, and sequence padding
- `src/model.py`: model architecture definition
- `src/artifacts.py`: save and load helpers for the model and tokenizer
- `src/inference.py`: reusable prediction logic used by the app
- `src/train.py`: training entry point
- `src/app.py`: Streamlit UI

## Installation

### 1. Clone or download the project

Make sure you are inside the project folder:

```powershell
cd "Sentiment Analysis"
```

### 2. Create a virtual environment

```powershell
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

## Requirements

The project currently depends on:

- TensorFlow
- pandas
- numpy
- scikit-learn
- streamlit

## Run the Project

### Train the model

If you want to retrain the model from the dataset:

```powershell
python -m src.train
```

This will generate or overwrite:

- `lstm_sentiment.h5`
- `tokenizer.pkl`

### Launch the Streamlit app

```powershell
streamlit run src/app.py
```

Then open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## Run with Docker

Build the image:

```powershell
docker build -t sentiment-analysis-app .
```

Run the container:

```powershell
docker run -p 8501:8501 sentiment-analysis-app
```

Then open:

```text
http://localhost:8501
```

## Run Tests

Run the basic preprocessing test suite with:

```powershell
python -m unittest discover -s tests
```

## Example Workflow

1. Install dependencies
2. Train the model if needed
3. Start the Streamlit app
4. Paste a movie review into the text box
5. Click `Analyze`
6. View the predicted sentiment and confidence score

## Files Explained

- `src/config.py`: shared hyperparameters and defaults
- `src/paths.py`: project filesystem locations
- `src/data.py`: dataset preparation helpers
- `src/preprocess.py`: cleaning, tokenization, and padding utilities
- `src/model.py`: neural network definition
- `src/artifacts.py`: saved artifact loading and persistence helpers
- `src/inference.py`: prediction logic shared by the app
- `src/train.py`: full training and model export pipeline
- `src/app.py`: Streamlit user interface for predictions
- `tests/test_preprocess.py`: basic unit tests for text cleaning
- `data/IMDB_Dataset.csv`: source dataset
- `lstm_sentiment.h5`: saved trained model
- `tokenizer.pkl`: saved tokenizer used during training

## Notes and Limitations

- The app is focused on movie review sentiment, not general purpose sentiment analysis.
- The UI now shows both model confidence and positive-class probability.
- Training and inference still depend on TensorFlow being installed locally.
- The saved model and tokenizer are written to the project root for simplicity.
- The dependencies in `requirements.txt` are not pinned to exact versions.

## Future Improvements

- add a `README` badge and screenshots
- pin dependency versions
- add model evaluation metrics such as precision, recall, and F1-score
- add early stopping and model checkpoints
- improve preprocessing with out-of-vocabulary handling
- add tests and a more production friendly project layout

## Author

Created as a sentiment analysis project using TensorFlow, Keras, and Streamlit.
