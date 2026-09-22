import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, precision_score, recall_score, classification_report

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import load_config

FEATURES_PATH = os.path.expanduser('~/Desktop/mobitel project/Mobitel_Alarm_Fault_Prediction/telecom_features_july_2026.csv')
MODEL_SAVE_PATH = os.path.expanduser('~/Desktop/mobitel project/Mobitel_Alarm_Fault_Prediction/models/calibrated_xgboost_outage.pkl')
PREDICTIONS_OUTPUT_PATH = os.path.expanduser('~/Desktop/mobitel project/Mobitel_Alarm_Fault_Prediction/data/latest_predictions.csv')

def get_base_classifier(scale_pos_weight: float = 1.0):
    """Instantiates XGBoost Classifier with fallback to RandomForest if libomp is absent."""
    try:
        from xgboost import XGBClassifier
        print("Using XGBClassifier base model...")
        return XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric='logloss'
        )
    except Exception as e:
        print(f"XGBoost unavailable ({e}). Falling back to RandomForestClassifier...")
        return RandomForestClassifier(
            n_estimators=150,
            max_depth=6,
            class_weight='balanced',
            random_state=42
        )

def get_predictor_features(df: pd.DataFrame) -> list:
    """Returns all feature columns excluding metadata IDs and target variables."""
    metadata_cols = [
        'site_id', 'node_identifier', 'enodeb_id', 'gnodeb_id', 'window_timestamp',
        'alarm_id', 'mo_name', 'location_information', 'primary_entity'
    ]
    target_cols = [
        'target_outage_next_2h', 'target_cell_unavailable_2h', 'target_cell_outage_2h',
        'target_cell_fault_2h', 'target_service_unavailable_2h'
    ]
    exclude = set(metadata_cols + target_cols)
    return [col for col in df.columns if col not in exclude]

def train_xgboost_model(features_csv_path: str = FEATURES_PATH, model_save_path: str = MODEL_SAVE_PATH) -> dict:
    """
    Trains tuned XGBoost / RandomForest Classifier with probability calibration (CalibratedClassifierCV)
    syncing hyperparameters from notebooks. Saves model payload with metadata.
    """
    print(f"Loading feature matrix from {features_csv_path}...")
    df = pd.read_csv(features_csv_path)
    df['window_timestamp'] = pd.to_datetime(df['window_timestamp'])

    config = load_config()
    target_col = 'target_outage_next_2h'

    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in feature matrix.")

    predictor_cols = get_predictor_features(df)
    print(f"Discovered Predictor Features ({len(predictor_cols)}): {predictor_cols}")

    # Chronological Split (Train: < July 23, Test: >= July 23)
    split_date = pd.Timestamp('2026-07-23 00:00:00')
    train_df = df[df['window_timestamp'] < split_date].copy()
    test_df = df[df['window_timestamp'] >= split_date].copy()

    # Fallback to 80/20 split if timestamp split leaves either set empty
    if train_df.empty or test_df.empty:
        shuffled = df.sample(frac=1.0, random_state=42)
        split_idx = int(len(shuffled) * 0.8)
        train_df = shuffled.iloc[:split_idx]
        test_df = shuffled.iloc[split_idx:]

    X_train, y_train = train_df[predictor_cols], train_df[target_col]
    X_test, y_test = test_df[predictor_cols], test_df[target_col]

    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    scale_pos_weight = float(neg_count / max(pos_count, 1))

    print(f"Train samples: {len(X_train)} (Outage Rate: {y_train.mean():.4f}) | Scale Pos Weight: {scale_pos_weight:.2f}")

    base_clf = get_base_classifier(scale_pos_weight=scale_pos_weight)

    print("Training model with CalibratedClassifierCV (Sigmoid)...")
    calibrated_clf = CalibratedClassifierCV(
        estimator=base_clf,
        method='sigmoid',
        cv=3
    )
    calibrated_clf.fit(X_train, y_train)

    # Performance Evaluation
    if not X_test.empty and y_test.sum() > 0:
        y_pred_proba = calibrated_clf.predict_proba(X_test)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall, precision)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        best_idx = np.argmax(f1_scores)
        optimal_thresh = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
    else:
        pr_auc, roc_auc, optimal_thresh = 0.0, 0.0, 0.5

    payload = {
        'model': calibrated_clf,
        'features': predictor_cols,
        'optimal_threshold': optimal_thresh,
        'pr_auc': pr_auc,
        'roc_auc': roc_auc,
        'active_targets': config.get('ACTIVE_TARGETS', []),
        'active_features': config.get('ACTIVE_FEATURES', []),
        'environment_mode': config.get('ENVIRONMENT_MODE', 'STAGING')
    }

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(payload, model_save_path)
    print(f"Model saved successfully to {model_save_path} (Optimal Threshold: {optimal_thresh:.4f})")
    return payload

def run_model_inference(features_csv_path: str = FEATURES_PATH, model_path: str = MODEL_SAVE_PATH, output_predictions_path: str = PREDICTIONS_OUTPUT_PATH) -> pd.DataFrame:
    """
    Inference Engine:
    Reads feature matrix, loads model, dynamically aligns features to prevent feature mismatch errors,
    and writes predictions to latest_predictions.csv.
    """
    if not os.path.exists(features_csv_path):
        raise FileNotFoundError(f"Features file missing at {features_csv_path}")

    if not os.path.exists(model_path):
        print(f"Model file missing at {model_path}. Training new model...")
        train_xgboost_model(features_csv_path=features_csv_path, model_save_path=model_path)

    payload = joblib.load(model_path)
    if isinstance(payload, dict) and 'model' in payload:
        model = payload['model']
        expected_features = payload.get('features', [])
        threshold = payload.get('optimal_threshold', 0.5)
    else:
        model = payload
        expected_features = get_predictor_features(pd.read_csv(features_csv_path, nrows=5))
        threshold = 0.5

    df_feats = pd.read_csv(features_csv_path)

    # Dynamic Feature Alignment to handle NOC Admin feature toggling without breaking
    X_input = pd.DataFrame(index=df_feats.index)
    for col in expected_features:
        if col in df_feats.columns:
            X_input[col] = df_feats[col]
        else:
            X_input[col] = 0.0

    probas = model.predict_proba(X_input)[:, 1]
    preds = (probas >= threshold).astype(int)

    result_df = df_feats.copy()
    result_df['outage_probability'] = probas
    result_df['predicted_outage'] = preds
    result_df['risk_level'] = np.where(probas >= 0.7, 'CRITICAL', np.where(probas >= 0.4, 'WARNING', 'NORMAL'))

    os.makedirs(os.path.dirname(output_predictions_path), exist_ok=True)
    result_df.to_csv(output_predictions_path, index=False)
    print(f"Inference complete: {len(result_df)} predictions generated -> {output_predictions_path}")
    return result_df

if __name__ == '__main__':
    train_xgboost_model()
    run_model_inference()
