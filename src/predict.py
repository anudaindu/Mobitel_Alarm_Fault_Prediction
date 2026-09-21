import os
import pandas as pd
import numpy as np
import joblib

def predict_cell_unavailability(feature_input: pd.DataFrame, model_path: str) -> np.ndarray:
    """
    Loads trained model and predicts probabilities for cell unavailability outage in the next 2 hours.
    """
    model = joblib.load(model_path)
    feature_cols = ['total_alarms_6h', 'critical_alarms_6h', 'major_alarms_6h', 'rru_alarms_6h', 'bbu_alarms_6h', 'total_alarms_24h']
    X_input = feature_input[feature_cols]
    probabilities = model.predict_proba(X_input)[:, 1]
    return probabilities

if __name__ == '__main__':
    model_path = os.path.expanduser('~/Desktop/mobitel project/Mobitel_Alarm_Fault_Prediction/models/outage_prediction_rf.pkl')
    sample_df = pd.DataFrame([{
        'total_alarms_6h': 15,
        'critical_alarms_6h': 4,
        'major_alarms_6h': 6,
        'rru_alarms_6h': 3,
        'bbu_alarms_6h': 2,
        'total_alarms_24h': 45
    }])
    prob = predict_cell_unavailability(sample_df, model_path)[0]
    print(f"Sample Prediction Outage Probability (Next 2h): {prob:.4f}")
