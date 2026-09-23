import os
import sys
from datetime import datetime, timedelta

from airflow import DAG
# pyrefly: ignore [missing-import]
from airflow.decorators import task

# Define dynamic Base Directory (Project Root: Mobitel_Alarm_Fault_Prediction/)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

default_args = {
    'owner': 'mobitel_noc_team',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='mobitel_outage_pipeline',
    default_args=default_args,
    description='Mobitel 4G/5G Infrastructure Outage Prediction & Auto-Purge Pipeline',
    schedule='@daily',
    catchup=False,
    tags=['mobitel', 'ml', 'outage_prediction', 'telecom', 'sqlite']
) as dag:

    @task
    def ingest_batch_logs():
        """
        Ingest Batch Logs Task:
        Parses daily uploaded files, executes strict 6-character Site ID regex matching,
        categorizes enumerated alarms, and appends clean records to SQLite.
        """
        from src.preprocessing import scan_and_create_subset
        from src.storage_manager import insert_cleaned_alarms
        import pandas as pd

        # Dynamic paths relative to workspace root
        subset_csv = os.path.join(BASE_DIR, 'data', 'july_2026_subset_dataset.csv')
        raw_downloads = os.path.join(BASE_DIR, 'data', 'daily_uploads')

        if os.path.exists(subset_csv):
            df_cleaned = pd.read_csv(subset_csv, low_memory=False)
        elif os.path.exists(raw_downloads) and len(os.listdir(raw_downloads)) > 0:
            df_cleaned = scan_and_create_subset(raw_downloads, subset_csv)
        else:
            print("No raw log files or subset CSV found. Batch ingestion skipped.")
            return "Skipped"

        inserted = insert_cleaned_alarms(df_cleaned, file_source_name='airflow_batch_feed')
        print(f"Ingested and cleaned {inserted} alarm records into SQLite master history database.")
        return f"Ingested {inserted} records into SQLite"

    @task
    def auto_purge_60_days():
        """
        Auto-Purge 60-Days Task:
        Executes SQL query: DELETE FROM master_alarms WHERE datetime(event_time) < datetime('now', '-60 days')
        """
        from src.storage_manager import purge_expired_logs
        purge_res = purge_expired_logs(retention_days=60)
        print(f"Auto-purge complete: {purge_res}")
        return purge_res

    @task
    def run_model_inference():
        """
        Run Model Inference Task:
        Evaluates active features and targets from config/settings.json,
        generates rolling aggregations (6h, 24h, 14d, 30d) at the site_id level,
        and updates data/latest_predictions.csv.
        """
        from src.feature_engineering import generate_sliding_window_features
        from src.model import run_model_inference as execute_inference

        subset_csv = os.path.join(BASE_DIR, 'data', 'july_2026_subset_dataset.csv')
        features_csv = os.path.join(BASE_DIR, 'data', 'telecom_features_july_2026.csv')
        predictions_output = os.path.join(BASE_DIR, 'data', 'latest_predictions.csv')

        print("1. Generating config-driven sliding window features...")
        generate_sliding_window_features(subset_csv, features_csv)

        print("2. Executing model inference pipeline...")
        preds_df = execute_inference(features_csv_path=features_csv, output_predictions_path=predictions_output)

        outage_count = int(preds_df['predicted_outage'].sum())
        total_sites = preds_df['site_id'].nunique()
        print(f"Inference complete: {outage_count} high-risk outage warnings generated across {total_sites} towers.")
        return f"Inference complete for {total_sites} towers"

    # Define Airflow task execution sequence
    task_ingest = ingest_batch_logs()
    task_purge = auto_purge_60_days()
    task_infer = run_model_inference()

    task_ingest >> task_purge >> task_infer