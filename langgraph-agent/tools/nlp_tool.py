import requests
from langchain_core.tools import tool

@tool
def analyze_text_for_risk(text_to_analyze: str) -> dict:
    """
    Analyzes a piece of text (like a merchant name or transaction description) 
    for potential risk using a sentiment analysis model. Returns a risk score 
    between 0.0 (low risk) and 1.0 (high risk). Use this to assess non-numeric 
    information.
    """
    # The URL of the NLP API you created
    api_url = "http://localhost:8001/nlp-score"
    
    # The data to send
    payload = {"text": text_to_analyze}
    
    print(f"DEBUG: Calling NLP API with payload: {payload}")
    
    try:
        response = requests.post(api_url, json=payload)
        response.raise_for_status() # Raise an exception for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"NLP API call failed: {e}"}

