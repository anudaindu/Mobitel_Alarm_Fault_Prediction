import os
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.decorators import task

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
    tags=['mobitel', 'ml', 'outage_prediction', 'telecom']
) as dag:

    @task
    def ingest_and_clean_data():
        """Reads daily raw alarm logs, applies fallback parsing, and stores records in SQLite DB."""
        from src.preprocessing import scan_and_create_subset
        from src.storage_manager import insert_cleaned_alarms
        import pandas as pd

        subset_csv = os.path.expanduser('~/Desktop/mobitel project/Mobitel_Alarm_Fault_Prediction/july_2026_subset_dataset.csv')
        raw_downloads = os.path.expanduser('~/Downloads/AlarmLogs20260803113740030')

        if os.path.exists(subset_csv):
            df_cleaned = pd.read_csv(subset_csv, low_memory=False)
        elif os.path.exists(raw_downloads):
            df_cleaned = scan_and_create_subset(raw_downloads, subset_csv)
        else:
            print("No raw log files or subset CSV found. Ingestion skipped.")
            return "Skipped"

        inserted = insert_cleaned_alarms(df_cleaned, file_source_name='daily_ingestion_feed')
        print(f"Ingested and cleaned {inserted} alarm records into master history database.")
        return f"Ingested {inserted} records"

    @task
    def auto_purge_60_days():
        """Executes automated 60-day retention auto-purge policy SQL query on master database."""
        from src.storage_manager import purge_expired_logs
        purge_res = purge_expired_logs(retention_days=60)
        print(f"Auto-purge complete: {purge_res}")
        return purge_res

    @task
    def run_model_inference():
        """Generates dynamic feature matrix based on config/settings.json and outputs predictions."""
        from src.feature_engineering import generate_sliding_window_features
        from src.model import run_model_inference as execute_inference

        subset_csv = os.path.expanduser('~/Desktop/mobitel project/Mobitel_Alarm_Fault_Prediction/july_2026_subset_dataset.csv')
        features_csv = os.path.expanduser('~/Desktop/mobitel project/Mobitel_Alarm_Fault_Prediction/telecom_features_july_2026.csv')
        predictions_output = os.path.expanduser('~/Desktop/mobitel project/Mobitel_Alarm_Fault_Prediction/data/latest_predictions.csv')

        print("1. Generating config-driven sliding window features...")
        generate_sliding_window_features(subset_csv, features_csv)

        print("2. Executing model inference pipeline...")
        preds_df = execute_inference(features_csv_path=features_csv, output_predictions_path=predictions_output)

        outage_count = int(preds_df['predicted_outage'].sum())
        total_sites = preds_df['site_id'].nunique()
        print(f"Inference complete: {outage_count} high-risk outage warnings generated across {total_sites} towers.")
        return f"Inference complete for {total_sites} towers"

    # Define DAG task execution flow
    task_ingest = ingest_and_clean_data()
    task_purge = auto_purge_60_days()
    task_infer = run_model_inference()

    task_ingest >> task_purge >> task_infer
