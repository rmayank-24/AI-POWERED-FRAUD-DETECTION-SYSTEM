from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
from datetime import datetime
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# --- Load all models and the SHAP explainer ---
try:
    # We load the random_forest model to get the feature names, as all models were trained on the same features.
    model_for_features = joblib.load('models/random_forest.pkl')
    MODEL_FEATURES = model_for_features.feature_names_in_
    
    # Load the models that will be used for prediction
    iso_forest = joblib.load('models/isolation_forest.pkl')
    xgb = joblib.load('models/xgboost.pkl')
    shap_explainer = joblib.load('models/shap_explainer.pkl')
    
    logger.info("All models and explainers loaded successfully.")
except FileNotFoundError as e:
    logger.error(f"Model file not found: {e}. Please ensure all model files are in a 'models/' directory.")
    exit()
except Exception as e:
    logger.error(f"An error occurred during model loading: {e}")
    exit()

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_transaction():
    data = request.json
    
    # --- FIX: Re-create the full feature engineering logic from the original app ---
    # Use some sensible defaults for customer stats that would normally come from a database
    cust_stats = {
        'AvgAmount': 150.0, 'StdAmount': 75.0, 'MaxAmount': 1000.0,
        'AvgDuration': 120.0, 'UniqueLocations': 3
    }
    
    # Safely parse dates
    try:
        transaction_date = datetime.strptime(data['TransactionDate'], '%Y-%m-%dT%H:%M')
        prev_date = datetime.strptime(data['PreviousTransactionDate'], '%Y-%m-%dT%H:%M')
        days_since_last = (transaction_date - prev_date).days
    except (ValueError, TypeError):
        days_since_last = 1 # Default to 1 day if dates are invalid

    transaction_duration = float(data.get('TransactionDuration', 60))
    if transaction_duration == 0:
        transaction_duration = 1 # Avoid division by zero

    # Build the complete feature dictionary that the models expect
    features_dict = {
        'TransactionAmount': float(data.get('TransactionAmount', 0)),
        'TransactionDuration': transaction_duration,
        'LoginAttempts': int(data.get('LoginAttempts', 1)),
        'AccountBalance': float(data.get('AccountBalance', 1000)),
        'DaysSinceLastTransaction': days_since_last,
        'TransactionSpeed': float(data.get('TransactionAmount', 0)) / transaction_duration,
        'AvgAmount': cust_stats['AvgAmount'],
        'StdAmount': cust_stats['StdAmount'],
        'MaxAmount': cust_stats['MaxAmount'],
        'AvgDuration': cust_stats['AvgDuration'],
        'UniqueLocations': cust_stats['UniqueLocations'],
        'AmountDeviation': (float(data.get('TransactionAmount', 0)) - cust_stats['AvgAmount']) / (cust_stats['StdAmount'] if cust_stats['StdAmount'] != 0 else 1),
        'DurationDeviation': (transaction_duration - cust_stats['AvgDuration']) / (cust_stats['AvgDuration'] if cust_stats['AvgDuration'] != 0 else 1),
        'TransactionType': 0 if data.get('TransactionType', 'Debit') == 'Debit' else 1,
        'Location': hash(data.get('Location', '')) % 100,
        'DeviceID': hash(data.get('DeviceID', '')) % 100,
        'MerchantID': hash(data.get('MerchantID', '')) % 100,
        'Channel': {'ATM': 0, 'Online': 1, 'Branch': 2}.get(data.get('Channel', 'Online'), 1),
        'CustomerOccupation': {'Student': 0, 'Doctor': 1, 'Engineer': 2, 'Retired': 3}.get(data.get('CustomerOccupation', 'Engineer'), 2),
        'CustomerAge': int(data.get('CustomerAge', 30))
    }
    
    # Convert to DataFrame with columns in the correct order
    X = pd.DataFrame([features_dict], columns=MODEL_FEATURES)
    
    # Get predictions
    iso_score = -iso_forest.decision_function(X)[0]
    xgb_prob = xgb.predict_proba(X)[0, 1]
    
    # Calculate SHAP values
    shap_values_list = shap_explainer.shap_values(X)
    fraud_shap_values = shap_values_list[1][0]
    base_value = shap_explainer.expected_value[1]
    
    composite_score = (iso_score * 0.5 + xgb_prob * 0.5)
    
    return jsonify({
        'composite_score': float(composite_score),
        'xgboost_probability': float(xgb_prob),
        'shap_base_value': base_value,
        'shap_values': fraud_shap_values.tolist(),
        'feature_names': MODEL_FEATURES.tolist(),
        'feature_values': X.iloc[0].values.tolist()
    })

if __name__ == '__main__':
    os.makedirs("models", exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5050)
