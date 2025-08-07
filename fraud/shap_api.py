import joblib
import pandas as pd
import shap
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Dict, List

# Initialize the FastAPI app
app = FastAPI(
    title="SHAP Explanation API",
    description="An API to provide SHAP values for fraud detection model predictions.",
    version="1.0.0"
)

# --- Define the input data structure (must match the model's expected features) ---
class Transaction(BaseModel):
    TransactionAmount: float = Field(..., example=125.50)
    TransactionHour: int = Field(..., example=14)
    TransactionDay: int = Field(..., example=3)
    TransactionMonth: int = Field(..., example=8)
    TransactionYear: int = Field(..., example=2025)
    TransactionDuration: int = Field(..., example=300)
    CustomerAge: int = Field(..., example=35)
    AccountBalance: float = Field(..., example=5000.0)
    LoginAttempts: int = Field(..., example=1)
    PurchaseFrequency: int = Field(..., example=5)

# --- Load the Model and the SHAP Explainer ---
try:
    # Load the pre-trained model
    model = joblib.load('models/random_forest.pkl')
    print("Model models/random_forest.pkl loaded successfully.")
    
    # Load the pre-calculated SHAP explainer
    explainer = joblib.load('models/shap_explainer.pkl')
    print("SHAP explainer models/shap_explainer.pkl loaded successfully.")

    # Get the expected feature names from the model
    MODEL_FEATURES = model.feature_names_in_

except Exception as e:
    print(f"Error loading model or explainer: {e}")
    model = None
    explainer = None
    MODEL_FEATURES = []

# --- Define the SHAP Explanation Endpoint ---
@app.post("/shap_explain")
def get_shap_explanation(transaction: Transaction):
    """
    Receives transaction data and returns the SHAP values to explain the model's prediction.
    """
    if model is None or explainer is None:
        return {"error": "Model or explainer not loaded"}

    # Convert input to a DataFrame with columns in the correct order
    input_data = transaction.dict()
    input_df = pd.DataFrame([input_data], columns=MODEL_FEATURES)
    
    # Calculate SHAP values for the single prediction
    shap_values = explainer.shap_values(input_df)
    
    # For classification, shap_values is a list [class_0_values, class_1_values]
    # We are interested in the explanation for the "fraud" class (class 1)
    fraud_shap_values = shap_values[1][0]
    
    return {
        "base_value": explainer.expected_value[1],
        "shap_values": fraud_shap_values.tolist(),
        "feature_names": MODEL_FEATURES.tolist(),
        "feature_values": input_df.iloc[0].tolist()
    }

@app.get("/")
def read_root():
    return {"message": "SHAP Explanation API is running. Go to /docs for API documentation."}
