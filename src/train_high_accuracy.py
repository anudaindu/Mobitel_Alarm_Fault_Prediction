import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, precision_score, recall_score, classification_report
from sklearn.inspection import permutation_importance

FEATURES_PATH = os.path.expanduser('~/Desktop/mobitel project/Mobitel_Alarm_Fault_Prediction/telecom_features_july_2026.csv')
MODEL_SAVE_PATH = os.path.expanduser('~/Desktop/mobitel project/Mobitel_Alarm_Fault_Prediction/models/calibrated_xgboost_outage.pkl')
FIGURES_DIR = os.path.expanduser('~/Desktop/mobitel project/Mobitel_Alarm_Fault_Prediction/figures')

def train_and_calibrate_model():
    print(f"Loading features from {FEATURES_PATH}...")
    df = pd.read_csv(FEATURES_PATH)
    df['window_timestamp'] = pd.to_datetime(df['window_timestamp'])

    # Strict Chronological Split
    # Training Window: July 1 - July 22, 2026
    # Holdout Test Window: July 23 - July 31, 2026
    split_date = pd.Timestamp('2026-07-23 00:00:00')
    train_df = df[df['window_timestamp'] < split_date].copy()
    test_df = df[df['window_timestamp'] >= split_date].copy()

    feature_cols = ['total_alarms_6h', 'critical_alarms_6h', 'major_alarms_6h', 'rru_alarms_6h', 'bbu_alarms_6h', 'total_alarms_24h']
    target_col = 'target_outage_next_2h'

    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]

    print(f"Train samples: {len(X_train)} (July 1-22), Outage Rate: {y_train.mean():.4f}")
    print(f"Test samples:  {len(X_test)} (July 23-31), Outage Rate: {y_test.mean():.4f}")

    # Base Classifier
    base_clf = RandomForestClassifier(
        n_estimators=150,
        max_depth=6,
        class_weight='balanced',
        random_state=42
    )

    # Calibrated Classifier CV
    print("Training and calibrating probabilities using CalibratedClassifierCV...")
    calibrated_clf = CalibratedClassifierCV(
        estimator=base_clf,
        method='sigmoid',
        cv=3
    )
    calibrated_clf.fit(X_train, y_train)

    # Evaluate on Holdout Test Set
    y_pred_proba = calibrated_clf.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)

    print("\n=== CALIBRATED MODEL HOLDOUT TEST METRICS ===")
    print(f"PR-AUC:                {pr_auc:.4f}")
    print(f"ROC-AUC:               {roc_auc:.4f}")
    print(f"Precision (Thresh 0.5): {prec:.4f}")
    print(f"Recall (Thresh 0.5):    {rec:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Cell Unavailable']))

    # Permutation Feature Importance
    perm = permutation_importance(calibrated_clf, X_test, y_test, scoring='average_precision', n_repeats=10, random_state=42)
    imp_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': perm.importances_mean
    }).sort_values('Importance', ascending=False)

    print("\nFeature Importance Scores:")
    print(imp_df.to_string(index=False))

    # Save Calibrated Model Artifact
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    joblib.dump(calibrated_clf, MODEL_SAVE_PATH)
    print(f"\nCalibrated model artifact saved successfully to {MODEL_SAVE_PATH}")

    return calibrated_clf, pr_auc, roc_auc, prec, rec

if __name__ == '__main__':
    train_and_calibrate_model()
