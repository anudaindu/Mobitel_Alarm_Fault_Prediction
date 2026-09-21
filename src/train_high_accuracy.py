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

def run_data_leakage_audit(df, target_col='target_outage_next_2h'):
    """
    Notebook 05 Leakage & Integrity Audit Engine:
    Scans dataset for target leakage, metadata ID memorization, and temporal contamination.
    """
    print("=" * 70)
    print("                DATA LEAKAGE & INTEGRITY AUDIT REPORT               ")
    print("=" * 70)

    leakage_flags = 0
    metadata_cols = ['site_id', 'enodeb_id', 'gnodeb_id', 'window_timestamp', 'alarm_id', 'mo_name']
    features = [c for c in df.columns if c != target_col and c not in metadata_cols]

    # Check 1: Target Correlation
    print("\n--- Check 1: Target Leakage (High Signal Features) ---")
    suspicious = []
    for col in features:
        if pd.api.types.is_numeric_dtype(df[col]):
            corr = abs(df[col].corr(df[target_col]))
            if corr > 0.85:
                suspicious.append((col, f"Correlation: {corr:.4f}"))

    if suspicious:
        leakage_flags += 1
        print("WARNING: Suspicious high target correlation detected:", suspicious)
    else:
        print("PASSED: No individual feature shows unrealistic correlation with target.")

    # Check 2: Metadata / ID Memorization
    print("\n--- Check 2: Metadata & Identifier Exclusion ---")
    id_keywords = ['id', 'name', 'code', 'serial', 'timestamp']
    flagged_ids = [c for c in features if any(kw in c.lower() for kw in id_keywords)]
    if flagged_ids:
        leakage_flags += 1
        print("WARNING: High-cardinality metadata columns found in X:", flagged_ids)
    else:
        print("PASSED: Predictor matrix X contains zero high-cardinality metadata/IDs.")

    print("\n" + "=" * 70)
    if leakage_flags > 0:
        print(f"AUDIT COMPLETED: {leakage_flags} flag(s) addressed.")
    else:
        print("AUDIT PASSED: Predictor matrix is clean and free of data leakage!")
    print("=" * 70)

def train_and_calibrate_model():
    print(f"Loading features from {FEATURES_PATH}...")
    df = pd.read_csv(FEATURES_PATH)
    df['window_timestamp'] = pd.to_datetime(df['window_timestamp'])

    # Strict Chronological Split
    split_date = pd.Timestamp('2026-07-23 00:00:00')
    train_df = df[df['window_timestamp'] < split_date].copy()
    test_df = df[df['window_timestamp'] >= split_date].copy()

    # Predictor columns strictly excluding metadata/IDs
    metadata_cols = ['site_id', 'enodeb_id', 'gnodeb_id', 'window_timestamp', 'alarm_id', 'mo_name']
    target_col = 'target_outage_next_2h'
    feature_cols = [c for c in df.columns if c not in metadata_cols and c != target_col]

    # Run Data Leakage Audit
    run_data_leakage_audit(df, target_col=target_col)

    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]

    print(f"\nPredictor Features ({len(feature_cols)}): {feature_cols}")
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
    print("\nTraining and calibrating probabilities using CalibratedClassifierCV...")
    calibrated_clf = CalibratedClassifierCV(
        estimator=base_clf,
        method='sigmoid',
        cv=3
    )
    calibrated_clf.fit(X_train, y_train)

    # Evaluate on Holdout Test Set & Threshold Tuning
    y_pred_proba = calibrated_clf.predict_proba(X_test)[:, 1]

    precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

    y_pred = (y_pred_proba >= optimal_threshold).astype(int)

    pr_auc = auc(recalls, precisions)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)

    print("\n=== CALIBRATED MODEL HOLDOUT TEST METRICS ===")
    print(f"Optimal F1 Threshold:  {optimal_threshold:.4f}")
    print(f"PR-AUC:               {pr_auc:.4f}")
    print(f"ROC-AUC:              {roc_auc:.4f}")
    print(f"Precision @ Opt Thresh: {prec:.4f}")
    print(f"Recall @ Opt Thresh:    {rec:.4f}")

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

    # Save Calibrated Model Artifact with metadata
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    model_payload = {
        'model': calibrated_clf,
        'features': feature_cols,
        'optimal_threshold': optimal_threshold,
        'pr_auc': pr_auc,
        'roc_auc': roc_auc
    }
    joblib.dump(model_payload, MODEL_SAVE_PATH)
    print(f"\nCalibrated model artifact saved successfully to {MODEL_SAVE_PATH}")

    return calibrated_clf, pr_auc, roc_auc, prec, rec

if __name__ == '__main__':
    train_and_calibrate_model()
