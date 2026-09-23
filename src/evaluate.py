import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, precision_score, recall_score, classification_report
from sklearn.inspection import permutation_importance

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def evaluate_outage_classifier(features_csv_path: str, model_save_path: str, figures_dir: str):
    """
    Evaluates trained model on July 23-31 holdout test set.
    Computes PR-AUC, ROC-AUC, Precision, Recall, and Permutation Feature Importance.
    """
    df = pd.read_csv(features_csv_path)
    df['window_timestamp'] = pd.to_datetime(df['window_timestamp'])

    split_date = pd.Timestamp('2026-07-23 00:00:00')
    test_df = df[df['window_timestamp'] >= split_date].copy()

    feature_cols = ['total_alarms_6h', 'critical_alarms_6h', 'major_alarms_6h', 'rru_alarms_6h', 'bbu_alarms_6h', 'total_alarms_24h']
    target_col = 'target_outage_next_2h'

    X_test, y_test = test_df[feature_cols], test_df[target_col]

    model = joblib.load(model_save_path)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)

    print("=== HOLDOUT TEST EVALUATION RESULTS (July 23-31, 2026) ===")
    print(f"PR-AUC:                {pr_auc:.4f}")
    print(f"ROC-AUC:               {roc_auc:.4f}")
    print(f"Precision (Thresh 0.5): {prec:.4f}")
    print(f"Recall (Thresh 0.5):    {rec:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Cell Unavailable']))

    perm = permutation_importance(model, X_test, y_test, scoring='average_precision', n_repeats=10, random_state=42)
    imp_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': perm.importances_mean
    }).sort_values('Importance', ascending=False)

    print("\nTop Predictive Alarm Features:")
    print(imp_df.to_string(index=False))

    os.makedirs(figures_dir, exist_ok=True)

    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, color='#1f77b4', lw=2, label=f'Model (PR-AUC = {pr_auc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (July 23-31 Test Holdout)')
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'pr_curve.png'), dpi=300)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    sns.barplot(data=imp_df, x='Importance', y='Feature', hue='Feature', palette='mako', legend=False)
    plt.title('Feature Importance (PR-AUC Permutation Impact)')
    plt.xlabel('PR-AUC Importance Score')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'feature_importance.png'), dpi=300)
    plt.close()

    metrics_dict = {
        'pr_auc': pr_auc,
        'roc_auc': roc_auc,
        'precision': prec,
        'recall': rec,
        'feature_importance': imp_df.to_dict(orient='records')
    }
    return metrics_dict

if __name__ == '__main__':
    features_path = os.path.join(BASE_DIR, 'data', 'telecom_features_july_2026.csv')
    model_path = os.path.join(BASE_DIR, 'models', 'calibrated_xgboost_outage.pkl')
    fig_dir = os.path.join(BASE_DIR, 'figures')
    evaluate_outage_classifier(features_path, model_path, fig_dir)
