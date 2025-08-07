from langchain_core.tools import tool
import requests

# --- FIX: Updated the tool to accept and send the CORRECT feature names ---
@tool
def check_transaction_for_fraud(
    TransactionAmount: float, 
    CustomerAge: int, 
    AccountBalance: float, 
    LoginAttempts: int
) -> dict:
    """
    Analyzes a financial transaction for potential fraud using a specialized machine learning model.
    You must provide the TransactionAmount, CustomerAge, AccountBalance, and LoginAttempts.
    Returns a dictionary containing the fraud prediction and probability score.
    """
    api_url = "http://localhost:8000/predict"
    
    # Create the payload with the correct feature names.
    # We provide default values for the features not included in the function signature.
    payload = {
        "TransactionAmount": TransactionAmount,
        "TransactionHour": 12,  # Default value
        "TransactionDay": 1,    # Default value
        "TransactionMonth": 1,  # Default value
        "TransactionYear": 2024,# Default value
        "TransactionDuration": 120, # Default value
        "CustomerAge": CustomerAge,
        "AccountBalance": AccountBalance,
        "LoginAttempts": LoginAttempts,
        "PurchaseFrequency": 1 # Default value
    }
    
    print(f"DEBUG: Calling Fraud API with payload: {payload}")
    
    try:
        response = requests.post(api_url, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"API call failed: {e}"}
