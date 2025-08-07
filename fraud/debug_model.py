#!/usr/bin/env python3
import joblib
import os

def check_model():
    """Check what features the XGBoost model expects"""
    try:
        model_path = 'models/xgboost.pkl'
        if not os.path.exists(model_path):
            print(f"ERROR: {model_path} does not exist!")
            return
        
        model = joblib.load(model_path)
        features = model.feature_names_in_
        
        print("=== XGBoost Model Features ===")
        print(f"Number of features: {len(features)}")
        print("Features (in order):")
        for i, feature in enumerate(features):
            print(f"  {i+1}. {feature}")
            
    except Exception as e:
        print(f"Error loading model: {e}")

if __name__ == "__main__":
    check_model()
