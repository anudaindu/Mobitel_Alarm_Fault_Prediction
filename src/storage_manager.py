import os
import glob
import sqlite3
import datetime
import pandas as pd

DATA_DIR = os.path.expanduser('~/Desktop/mobitel project/Mobitel_Alarm_Fault_Prediction/data')
UPLOADS_DIR = os.path.join(DATA_DIR, 'daily_uploads')
DB_PATH = os.path.join(DATA_DIR, 'telecom_master_history.db')

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

def get_db_connection() -> sqlite3.Connection:
    """Initializes and returns a pure Python sqlite3 connection to telecom_master_history.db."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS master_alarms (
            site_id TEXT,
            node_identifier TEXT,
            event_time TEXT,
            alarm_name TEXT,
            severity TEXT,
            duration_minutes REAL,
            file_source TEXT,
            ingestion_time TEXT
        )
    """)
    return conn

def insert_cleaned_alarms(df: pd.DataFrame, file_source_name: str) -> int:
    """Inserts a cleaned alarm DataFrame into the SQLite master database."""
    if df.empty:
        return 0

    conn = get_db_connection()
    site_col = df['site_id'].astype(str) if 'site_id' in df.columns else '-'
    node_col = df['node_identifier'].astype(str) if 'node_identifier' in df.columns else '-'

    df_to_insert = pd.DataFrame({
        'site_id': site_col,
        'node_identifier': node_col,
        'event_time': pd.to_datetime(df['event_time']).dt.strftime('%Y-%m-%d %H:%M:%S'),
        'alarm_name': df['alarm_name'].astype(str),
        'severity': df['severity'].astype(str),
        'duration_minutes': df['duration_minutes'] if 'duration_minutes' in df.columns else 0.0,
        'file_source': file_source_name,
        'ingestion_time': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    })

    df_to_insert.to_sql('master_alarms', conn, if_exists='append', index=False)
    conn.close()

    inserted_count = len(df_to_insert)
    purge_expired_logs(retention_days=60)
    return inserted_count

def purge_expired_logs(retention_days: int = 60) -> dict:
    """
    Automated 60-Day Retention Auto-Purge Policy (Pure SQLite Engine):
    Executes SQL query: DELETE FROM master_alarms WHERE datetime(event_time) < datetime('now', '-60 days')
    And purges raw upload files older than 60 days.
    """
    conn = get_db_connection()
    rows_purged = 0

    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM master_alarms WHERE datetime(event_time) < datetime('now', '-60 days')")
        rows_purged = cursor.rowcount
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Warning: Exception during auto-purge SQL execution: {e}")

    # Scan raw upload files and remove files whose mtime is older than retention_days
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
                print(f"Failed deleting raw file {raw_file}: {e}")

    print(f"[SQLite Auto-Purge 60-Days] Purged {rows_purged} expired SQL database rows and {purged_files} raw files.")
    return {
        'purged_db_rows': rows_purged,
        'purged_raw_files': purged_files
    }

def get_master_history_summary() -> dict:
    """Returns total record count, min date, max date, and site count from SQLite master index."""
    conn = get_db_connection()
    summary = pd.read_sql_query("""
        SELECT 
            COUNT(*) as total_records,
            MIN(event_time) as min_date,
            MAX(event_time) as max_date,
            COUNT(DISTINCT site_id) as site_count
        FROM master_alarms
    """, conn)
    conn.close()

    if not summary.empty:
        return summary.iloc[0].to_dict()
    return {'total_records': 0, 'min_date': None, 'max_date': None, 'site_count': 0}

if __name__ == '__main__':
    print("Testing Pure SQLite Storage Manager & Auto-Purge 60-Days Engine...")
    purge_expired_logs(retention_days=60)
    print("SQLite Master History Summary:", get_master_history_summary())
