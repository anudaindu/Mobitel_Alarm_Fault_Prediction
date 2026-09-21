import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

def train_outage_classifier(features_csv_path: str, model_save_path: str):
    """
    Trains tabular classification models using a strict chronological split:
    - Training Set: July 1 - July 22
    - Testing Set: July 23 - July 31
    Handles class imbalance and optimizes for PR-AUC.
    """
    df = pd.read_csv(features_csv_path)
    df['window_timestamp'] = pd.to_datetime(df['window_timestamp'])

    split_date = pd.Timestamp('2026-07-23 00:00:00')
    train_df = df[df['window_timestamp'] < split_date].copy()

    feature_cols = ['total_alarms_6h', 'critical_alarms_6h', 'major_alarms_6h', 'rru_alarms_6h', 'bbu_alarms_6h', 'total_alarms_24h']
    target_col = 'target_outage_next_2h'

    X_train, y_train = train_df[feature_cols], train_df[target_col]

    print(f"Training on {len(X_train)} samples (July 1-22). Outage proportion: {y_train.mean():.4f}")

    rf_model = RandomForestClassifier(
        n_estimators=150,
        max_depth=6,
        class_weight='balanced',
        random_state=42
    )
    rf_model.fit(X_train, y_train)

    os.makedirs(os.path.dirname(os.path.abspath(model_save_path)), exist_ok=True)
    joblib.dump(rf_model, model_save_path)
    print(f"Model successfully saved to {model_save_path}")
    return rf_model

if __name__ == '__main__':
    features_path = os.path.expanduser('~/Desktop/mobitel project/Mobitel_Alarm_Fault_Prediction/telecom_features_july_2026.csv')
    model_path = os.path.expanduser('~/Desktop/mobitel project/Mobitel_Alarm_Fault_Prediction/models/outage_prediction_rf.pkl')
    train_outage_classifier(features_path, model_path)
