import os
import sys
import shutil
import pandas as pd
import numpy as np
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import clean_raw_alarm_dataframe
from src.storage_manager import insert_cleaned_alarms, UPLOADS_DIR

MODEL_PATH = os.path.expanduser('~/Desktop/mobitel project/Mobitel_Alarm_Fault_Prediction/models/calibrated_xgboost_outage.pkl')

def process_manual_file_upload(file_input, original_filename: str) -> dict:
    """
    Processes a manually uploaded Excel (.xlsx/.xls) or CSV daily alarm log:
    1. Saves copy to data/daily_uploads/
    2. Runs integrated notebook data cleaning pipeline.
    3. Ingests cleaned records into DuckDB master database.
    4. Triggers automated 60-day retention auto-purge.
    5. Computes site sliding-window features and predicts 2-hour outage risk scores.
    """
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    saved_file_path = os.path.join(UPLOADS_DIR, original_filename)

    if isinstance(file_input, str):
        shutil.copy2(file_input, saved_file_path)
    else:
        with open(saved_file_path, 'wb') as f:
            f.write(file_input.getbuffer() if hasattr(file_input, 'getbuffer') else file_input.read())

    if original_filename.endswith('.csv'):
        df_raw = pd.read_csv(saved_file_path, low_memory=False)
    else:
        df_header_check = pd.read_excel(saved_file_path, header=None, nrows=12, engine='calamine')
        header_idx = 5
        for idx, row in df_header_check.iterrows():
            row_str = " ".join([str(v) for v in row.values if pd.notna(v)])
            if 'Severity' in row_str and ('Alarm ID' in row_str or 'Occurred' in row_str or 'MO Name' in row_str):
                header_idx = idx
                break
        df_raw = pd.read_excel(saved_file_path, skiprows=header_idx, engine='calamine')

    cleaned_df = clean_raw_alarm_dataframe(df_raw)
    inserted_records = insert_cleaned_alarms(cleaned_df, original_filename)

    cleaned_df['is_critical'] = (cleaned_df['severity'].astype(str).str.lower() == 'critical').astype(int)
    cleaned_df['is_major'] = (cleaned_df['severity'].astype(str).str.lower() == 'major').astype(int)

    alarm_str = cleaned_df['alarm_name'].astype(str).str.lower()
    cleaned_df['is_rru'] = alarm_str.str.contains(r'rf|rru|vswr', regex=True).astype(int)
    cleaned_df['is_bbu'] = alarm_str.str.contains(r'bbu|board|subrack|transmission', regex=True).astype(int)

    sites = cleaned_df['site_id'].unique()
    latest_t = cleaned_df['event_time'].max()
    
    feature_rows = []
    for site in sites:
        site_df = cleaned_df[cleaned_df['site_id'] == site]
        times = site_df['event_time'].values
        is_crit = site_df['is_critical'].values
        is_maj = site_df['is_major'].values
        is_rru = site_df['is_rru'].values
        is_bbu = site_df['is_bbu'].values

        t_np = latest_t.to_datetime64()
        start_6h = (latest_t - pd.Timedelta(hours=6)).to_datetime64()
        start_24h = (latest_t - pd.Timedelta(hours=24)).to_datetime64()

        idx_6h = (times >= start_6h) & (times <= t_np)
        idx_24h = (times >= start_24h) & (times <= t_np)

        feature_rows.append({
            'site_id': site,
            'window_timestamp': latest_t,
            'total_alarms_6h': int(np.sum(idx_6h)),
            'critical_alarms_6h': int(np.sum(is_crit[idx_6h])),
            'major_alarms_6h': int(np.sum(is_maj[idx_6h])),
            'rru_alarms_6h': int(np.sum(is_rru[idx_6h])),
            'bbu_alarms_6h': int(np.sum(is_bbu[idx_6h])),
            'total_alarms_24h': int(np.sum(idx_24h))
        })

    features_df = pd.DataFrame(feature_rows)

    model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
    feature_cols = ['total_alarms_6h', 'critical_alarms_6h', 'major_alarms_6h', 'rru_alarms_6h', 'bbu_alarms_6h', 'total_alarms_24h']
    
    if model is not None and not features_df.empty:
        probs = model.predict_proba(features_df[feature_cols])[:, 1]
        features_df['failure_probability'] = probs
        features_df['risk_status'] = features_df['failure_probability'].apply(
            lambda p: 'CRITICAL' if p >= 0.65 else ('WARNING' if p >= 0.35 else 'NOMINAL')
        )
        features_df.sort_values('failure_probability', ascending=False, inplace=True)
    else:
        features_df['failure_probability'] = 0.0
        features_df['risk_status'] = 'NOMINAL'

    return {
        'filename': original_filename,
        'raw_rows': len(df_raw),
        'cleaned_rows': len(cleaned_df),
        'inserted_db_records': inserted_records,
        'unique_sites_processed': len(sites),
        'latest_timestamp': latest_t,
        'site_risk_scores': features_df
    }

if __name__ == '__main__':
    print("Testing manual upload processor...")
    sample_file = os.path.expanduser('~/Downloads/AlarmLogs20260803113740030/AlarmLogs20260803113740030_1.xlsx')
    if os.path.exists(sample_file):
        res = process_manual_file_upload(sample_file, 'AlarmLogs20260803113740030_1.xlsx')
        print("Upload processing results:")
        print(f"File: {res['filename']} | Raw Rows: {res['raw_rows']} | Cleaned: {res['cleaned_rows']} | Ingested: {res['inserted_db_records']}")
        print(res['site_risk_scores'].head(5))
