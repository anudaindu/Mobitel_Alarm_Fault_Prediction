import os
import glob
import datetime
import pandas as pd

DATA_DIR = os.path.expanduser('~/Desktop/mobitel project/Mobitel_Alarm_Fault_Prediction/data')
UPLOADS_DIR = os.path.join(DATA_DIR, 'daily_uploads')
DB_PATH = os.path.join(DATA_DIR, 'telecom_master_history.db')

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

def get_db_connection():
    """Initializes and returns a SQLite or DuckDB connection to telecom_master_history.db."""
    try:
        import sqlite3
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
        return ('sqlite', conn)
    except Exception:
        import duckdb
        conn = duckdb.connect(DB_PATH)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS master_alarms (
                site_id VARCHAR,
                node_identifier VARCHAR,
                event_time TIMESTAMP,
                alarm_name VARCHAR,
                severity VARCHAR,
                duration_minutes DOUBLE,
                file_source VARCHAR,
                ingestion_time TIMESTAMP
            )
        """)
        return ('duckdb', conn)

def insert_cleaned_alarms(df: pd.DataFrame, file_source_name: str) -> int:
    """Inserts a cleaned alarm DataFrame into the master history database."""
    if df.empty:
        return 0

    db_type, conn = get_db_connection()
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

    if db_type == 'sqlite':
        df_to_insert.to_sql('master_alarms', conn, if_exists='append', index=False)
        conn.close()
    else:
        conn.execute("INSERT INTO master_alarms SELECT * FROM df_to_insert")
        conn.close()

    inserted_count = len(df_to_insert)
    purge_expired_logs(retention_days=60)
    return inserted_count

def purge_expired_logs(retention_days: int = 60) -> dict:
    """
    Automated 60-Day Retention Auto-Purge Policy:
    Runs SQL query: DELETE FROM master_alarms WHERE datetime(event_time) < datetime('now', '-60 days')
    And cleans up raw upload files older than retention_days.
    """
    db_type, conn = get_db_connection()
    rows_purged = 0

    try:
        if db_type == 'sqlite':
            cursor = conn.cursor()
            # Explicit SQL auto-purge query as per spec
            cursor.execute("DELETE FROM master_alarms WHERE datetime(event_time) < datetime('now', '-60 days')")
            rows_purged = cursor.rowcount
            conn.commit()
            conn.close()
        else:
            max_date_row = conn.execute("SELECT MAX(event_time) FROM master_alarms").fetchone()
            latest_date = max_date_row[0] if max_date_row and max_date_row[0] else pd.Timestamp.now()
            cutoff_date = pd.to_datetime(latest_date) - pd.Timedelta(days=retention_days)
            rows_purged = conn.execute("SELECT COUNT(*) FROM master_alarms WHERE event_time < ?", [cutoff_date]).fetchone()[0]
            if rows_purged > 0:
                conn.execute("DELETE FROM master_alarms WHERE event_time < ?", [cutoff_date])
            conn.close()
    except Exception as e:
        print(f"Warning: Exception during auto-purge SQL execution: {e}")

    # Scan raw upload files and remove files whose mtime is older than 60 days
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

    print(f"[Auto-Purge 60-Days] Purged {rows_purged} expired SQL database rows and {purged_files} raw files.")
    return {
        'purged_db_rows': rows_purged,
        'purged_raw_files': purged_files
    }

def get_master_history_summary() -> dict:
    """Returns summary of records stored in master alarms database."""
    db_type, conn = get_db_connection()
    if db_type == 'sqlite':
        summary = pd.read_sql_query("""
            SELECT 
                COUNT(*) as total_records,
                MIN(event_time) as min_date,
                MAX(event_time) as max_date,
                COUNT(DISTINCT site_id) as site_count
            FROM master_alarms
        """, conn)
        conn.close()
    else:
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
    print("Testing Storage Manager & Auto-Purge 60-Days Engine...")
    purge_expired_logs(retention_days=60)
    print("Master History Summary:", get_master_history_summary())
