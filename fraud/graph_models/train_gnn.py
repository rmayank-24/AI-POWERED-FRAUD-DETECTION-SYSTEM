from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# --- Load ONLY the XGBoost model and its SHAP explainer ---
try:
    xgb_model = joblib.load('models/xgboost.pkl')
    shap_explainer = joblib.load('models/shap_explainer.pkl')
    
    # Get the exact feature list required by this specific model
    MODEL_FEATURES = xgb_model.feature_names_in_
    
    logger.info("XGBoost model and SHAP explainer loaded successfully.")
except FileNotFoundError as e:
    logger.error(f"Model file not found: {e}. Please ensure 'xgboost.pkl' and 'shap_explainer.pkl' are in the 'models/' directory.")
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
    
    # --- Create a simple dictionary from the form data ---
    # This directly matches the features the model expects.
    features_dict = {
        'TransactionAmount': float(data.get('TransactionAmount', 0)),
        'TransactionDuration': float(data.get('TransactionDuration', 60)),
        'LoginAttempts': int(data.get('LoginAttempts', 1)),
        'AccountBalance': float(data.get('AccountBalance', 1000)),
        'DaysSinceLastTransaction': int(data.get('DaysSinceLastTransaction', 1))
    }
    
    # --- Create a DataFrame with only the features this model needs ---
    # We create a dictionary with default value 0 for any feature not in our simple form
    X_data = {key: [features_dict.get(key, 0)] for key in MODEL_FEATURES}
    X = pd.DataFrame(X_data, columns=MODEL_FEATURES)
    
    # --- Get prediction and SHAP values ---
    xgb_prob = xgb_model.predict_proba(X)[0, 1]
    
    shap_values_list = shap_explainer.shap_values(X)
    # The structure of shap_values can vary, we try to handle both list and single array cases
    fraud_shap_values = shap_values_list[1][0] if isinstance(shap_values_list, list) else shap_values_list[0]
    
    base_value = shap_explainer.expected_value
    # Handle case where expected_value is an array for multi-class classification
    if hasattr(base_value, "__len__"):
        base_value = base_value[1]

    return jsonify({
        'xgboost_probability': float(xgb_prob),
        'shap_base_value': base_value,
        'shap_values': fraud_shap_values.tolist(),
        'feature_names': MODEL_FEATURES.tolist(),
        'feature_values': X.iloc[0].values.tolist()
    })

if __name__ == '__main__':
    os.makedirs("models", exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5050)
