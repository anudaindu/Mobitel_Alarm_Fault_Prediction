import os
import glob
import datetime
import pandas as pd
import duckdb

DATA_DIR = os.path.expanduser('~/Desktop/mobitel project/Mobitel_Alarm_Fault_Prediction/data')
UPLOADS_DIR = os.path.join(DATA_DIR, 'daily_uploads')
DB_PATH = os.path.join(DATA_DIR, 'telecom_master_history.db')

os.makedirs(UPLOADS_DIR, exist_ok=True)

def get_db_connection():
    """Initializes and returns a DuckDB connection to telecom_master_history.db."""
    conn = duckdb.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS master_alarms (
            site_id VARCHAR,
            event_time TIMESTAMP,
            alarm_name VARCHAR,
            severity VARCHAR,
            duration_minutes DOUBLE,
            file_source VARCHAR,
            ingestion_time TIMESTAMP
        )
    """)
    return conn

def insert_cleaned_alarms(df: pd.DataFrame, file_source_name: str) -> int:
    """Inserts a cleaned alarm DataFrame into the DuckDB master storage database."""
    if df.empty:
        return 0

    conn = get_db_connection()
    df_to_insert = pd.DataFrame({
        'site_id': df['site_id'].astype(str),
        'event_time': pd.to_datetime(df['event_time']),
        'alarm_name': df['alarm_name'].astype(str),
        'severity': df['severity'].astype(str),
        'duration_minutes': df['duration_minutes'] if 'duration_minutes' in df.columns else 0.0,
        'file_source': file_source_name,
        'ingestion_time': pd.Timestamp.now()
    })

    conn.execute("INSERT INTO master_alarms SELECT * FROM df_to_insert")
    inserted_count = len(df_to_insert)
    conn.close()
    
    # Trigger automated 60-day retention auto-purge
    purge_expired_logs(retention_days=60)
    return inserted_count

def purge_expired_logs(retention_days: int = 60) -> dict:
    """
    Automated 60-Day Retention Auto-Purge Engine:
    Deletes raw upload files and database records where event_time < (latest_date - 60 days).
    """
    conn = get_db_connection()
    max_date_row = conn.execute("SELECT MAX(event_time) FROM master_alarms").fetchone()
    latest_date = max_date_row[0] if max_date_row and max_date_row[0] else pd.Timestamp.now()
    
    if isinstance(latest_date, str):
        latest_date = pd.to_datetime(latest_date)

    cutoff_date = latest_date - pd.Timedelta(days=retention_days)
    
    # Count rows to purge
    rows_to_purge = conn.execute("SELECT COUNT(*) FROM master_alarms WHERE event_time < ?", [cutoff_date]).fetchone()[0]
    
    if rows_to_purge > 0:
        conn.execute("DELETE FROM master_alarms WHERE event_time < ?", [cutoff_date])
        print(f"Purged {rows_to_purge} historical records older than {cutoff_date.strftime('%Y-%m-%d')} ({retention_days} days).")

    conn.close()

    # Scan raw upload files and remove files whose modification time is older than retention window
    purged_files = 0
    now_ts = datetime.datetime.now().timestamp()
    cutoff_seconds = retention_days * 86400

    for raw_file in glob.glob(os.path.join(UPLOADS_DIR, '*')):
        file_mtime = os.path.getmtime(raw_file)
        if (now_ts - file_mtime) > cutoff_seconds:
            try:
                os.remove(raw_file)
                purged_files += 1
                print(f"Purged expired raw upload file: {os.path.basename(raw_file)}")
            except Exception as e:
                print(f"Failed deleting file {raw_file}: {e}")

    return {
        'cutoff_date': cutoff_date,
        'purged_db_rows': rows_to_purge,
        'purged_raw_files': purged_files
    }

def get_master_history_summary() -> dict:
    """Returns total record count, min date, max date, and site count from DuckDB master index."""
    conn = get_db_connection()
    summary = conn.execute("""
        SELECT 
            COUNT(*) as total_records,
            MIN(event_time) as min_date,
            MAX(event_time) as max_date,
            COUNT(DISTINCT site_id) as site_count
        FROM master_alarms
    """).df()
    conn.close()
    if not summary.empty:
        return summary.iloc[0].to_dict()
    return {'total_records': 0, 'min_date': None, 'max_date': None, 'site_count': 0}

if __name__ == '__main__':
    print("Testing DuckDB Storage Manager & Auto-Purge Engine...")
    conn = get_db_connection()
    conn.close()
    print("DuckDB storage manager initialized successfully.")
    print("Current Master History Summary:", get_master_history_summary())
