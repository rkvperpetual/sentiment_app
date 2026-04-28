import os
from functools import lru_cache
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import pipeline


MODEL_NAME = os.getenv(
    "SENTIMENT_MODEL",
    "distilbert-base-uncased-finetuned-sst-2-english",
)


class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to analyze")


class PredictionResponse(BaseModel):
    label: str
    score: float


@lru_cache(maxsize=1)
def get_sentiment_pipeline() -> Any:
    return pipeline("sentiment-analysis", model=MODEL_NAME)


def parse_allowed_origins() -> list[str]:
    origins = os.getenv("ALLOWED_ORIGINS", "*")
    return [origin.strip() for origin in origins.split(",") if origin.strip()]


app = FastAPI(
    title="Sentiment Analysis API",
    version="1.0.0",
    description="FastAPI backend for transformer-based sentiment analysis.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=parse_allowed_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text must not be empty.")

    try:
        result = get_sentiment_pipeline()(text, truncation=True)[0]
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Prediction service unavailable.") from exc

    return PredictionResponse(
        label=str(result["label"]),
        score=round(float(result["score"]), 6),
    )
