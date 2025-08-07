from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import pipeline
from typing import Dict

app = FastAPI(title="NLP Risk Scoring API")

class TransactionText(BaseModel):
    text: str = Field(..., example="URGENT: Claim your prize now!")

try:
    sentiment_analyzer = pipeline(
        "sentiment-analysis", 
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )
    print("Sentiment analysis model loaded successfully.")
except Exception as e:
    print(f"Error loading NLP model: {e}")
    sentiment_analyzer = None

@app.post("/nlp-score", response_model=Dict[str, float])
def get_nlp_score(data: TransactionText):
    if sentiment_analyzer is None:
        return {"error": "NLP model not loaded", "risk_score": -1.0}

    results = sentiment_analyzer(data.text)
    sentiment = results[0]
    
    label = sentiment['label']
    score = sentiment['score']
    
    risk_score = 0.0
    if label == 'NEGATIVE':
        risk_score = score
    elif label == 'POSITIVE':
        risk_score = 1.0 - score

    return {
        "risk_score": risk_score,
        "sentiment_label": 1 if label == 'NEGATIVE' else 0,
        "sentiment_score": score
    }
