import os
import re
import glob
import pandas as pd
import numpy as np

def standardize_column_name(col_name: str) -> str:
    """Notebook cleaning routine: Standardizes column header formatting."""
    col_name = str(col_name).strip().lower()
    col_name = col_name.replace(' ', '_')
    col_name = re.sub(r'[^a-z0-9_]', '', col_name)
    return col_name

def parse_duration_minutes(duration_str) -> float:
    """Notebook cleaning routine: Parses duration string into numeric minutes."""
    if pd.isna(duration_str) or str(duration_str).strip() in ['-', '']:
        return 0.0
    val = str(duration_str).lower()
    hours, minutes, seconds = 0, 0, 0
    try:
        h_match = re.search(r'(\d+)\s*hour', val)
        m_match = re.search(r'(\d+)\s*min', val)
        s_match = re.search(r'(\d+)\s*sec', val)
        if h_match:
            hours = int(h_match.group(1))
        if m_match:
            minutes = int(m_match.group(1))
        if s_match:
            seconds = int(s_match.group(1))
        return float(hours * 60 + minutes + seconds / 60.0)
    except Exception:
        return 0.0

def clean_raw_alarm_dataframe(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Applies integrated notebook cleaning logic:
    1. Standardizes column header names.
    2. Maps eNodeB ID (4G) / gNodeB ID (5G) or parses Location Information for Site ID.
    3. Standardizes datetime (Occurred On).
    4. Deduplicates repeat alarm logs occurring within the same minute for the same site.
    5. Converts alarm duration to minutes.
    """
    df = df_raw.copy()
    df.columns = [standardize_column_name(c) for c in df.columns]

    # Find time column
    time_col = None
    for candidate in ['occurred_on_nt', 'occurred_on', 'event_time', 'occurred_on_st']:
        if candidate in df.columns:
            time_col = candidate
            break

    if not time_col:
        raise KeyError(f"No valid event time column found in DataFrame columns: {list(df.columns)}")

    df['event_time'] = pd.to_datetime(df[time_col], errors='coerce')
    df.dropna(subset=['event_time'], inplace=True)

    # Site ID Standardization (eNodeB ID / gNodeB ID / Location Info)
    site_id_col = None
    if 'enodeb_id' in df.columns:
        df['enodeb_clean'] = df['enodeb_id'].astype(str).str.strip()
        df['site_id'] = df['enodeb_clean'].apply(lambda x: None if x in ['-', 'nan', 'none', ''] else x)
    elif 'gnodeb_id' in df.columns:
        df['gnodeb_clean'] = df['gnodeb_id'].astype(str).str.strip()
        df['site_id'] = df['gnodeb_clean'].apply(lambda x: None if x in ['-', 'nan', 'none', ''] else x)
    else:
        df['site_id'] = None

    if 'location_information' in df.columns:
        df['site_id'] = df['site_id'].fillna(df['location_information'])
    elif 'alarm_source' in df.columns:
        df['site_id'] = df['site_id'].fillna(df['alarm_source'])
    elif 'mo_name' in df.columns:
        df['site_id'] = df['site_id'].fillna(df['mo_name'])

    df.dropna(subset=['site_id'], inplace=True)
    df['site_id'] = df['site_id'].astype(str).str.strip()

    # Standardize Alarm Name & Severity
    alarm_name_col = 'name' if 'name' in df.columns else ('alarm_name' if 'alarm_name' in df.columns else None)
    if alarm_name_col:
        df['alarm_name'] = df[alarm_name_col].astype(str)
    else:
        df['alarm_name'] = 'Unknown Alarm'

    if 'severity' in df.columns:
        df['severity'] = df['severity'].astype(str).str.capitalize()
    else:
        df['severity'] = 'Major'

    if 'alarm_duration' in df.columns:
        df['duration_minutes'] = df['alarm_duration'].apply(parse_duration_minutes)

    # Notebook Deduplication Routine: Deduplicate identical alarms for same site & same minute
    df['event_minute'] = df['event_time'].dt.floor('min')
    df.drop_duplicates(subset=['site_id', 'event_minute', 'alarm_name', 'severity'], inplace=True)
    df.drop(columns=['event_minute'], inplace=True, errors='ignore')

    df.sort_values(by=['site_id', 'event_time'], inplace=True)
    return df

def scan_and_create_subset(data_dir: str, output_csv_path: str) -> pd.DataFrame:
    """
    Scans Huawei Excel files, applies integrated notebook data cleaning,
    extracts top 20 outage towers + 15 healthy towers, and exports dataset.
    """
    files = sorted(glob.glob(os.path.join(data_dir, 'AlarmLogs*.xlsx')))
    if not files:
        raise FileNotFoundError(f"No AlarmLogs*.xlsx files found in {data_dir}")

    print(f"Scanning {len(files)} Huawei log files in {data_dir} with integrated notebook cleaning...")
    dfs = []
    for file_path in files:
        try:
            df_raw = pd.read_excel(file_path, header=None, nrows=12, engine='calamine')
            header_idx = 5
            for idx, row in df_raw.iterrows():
                row_str = " ".join([str(v) for v in row.values if pd.notna(v)])
                if 'Severity' in row_str and ('Alarm ID' in row_str or 'Occurred' in row_str or 'MO Name' in row_str):
                    header_idx = idx
                    break

            df = pd.read_excel(file_path, skiprows=header_idx, engine='calamine')
            cleaned_df = clean_raw_alarm_dataframe(df)
            dfs.append(cleaned_df)
        except Exception as e:
            print(f"Warning: Failed processing {os.path.basename(file_path)}: {e}")

    full_df = pd.concat(dfs, ignore_index=True)
    full_df['is_outage'] = full_df['alarm_name'].str.lower().str.contains('cell unavailable')

    # Select top 20 outage towers & 15 healthy towers
    site_outage_counts = full_df.groupby('site_id')['is_outage'].sum()
    top_20_outage_towers = site_outage_counts.sort_values(ascending=False).head(20).index.tolist()

    healthy_towers_series = site_outage_counts[site_outage_counts == 0]
    healthy_tower_activity = full_df[full_df['site_id'].isin(healthy_towers_series.index)]['site_id'].value_counts()
    healthy_15_towers = healthy_tower_activity.head(15).index.tolist()

    selected_35_towers = top_20_outage_towers + healthy_15_towers

    subset_df = full_df[full_df['site_id'].isin(selected_35_towers)].copy()
    subset_df.sort_values(by=['site_id', 'event_time'], inplace=True)

    os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)
    subset_df.to_csv(output_csv_path, index=False)
    print(f"Cleaned subset exported: {len(subset_df)} rows across 35 towers -> {output_csv_path}")
    return subset_df

if __name__ == '__main__':
    data_dir = os.path.expanduser('~/Downloads/AlarmLogs20260803113740030')
    output_path = os.path.expanduser('~/Desktop/mobitel project/Mobitel_Alarm_Fault_Prediction/july_2026_subset_dataset.csv')
    scan_and_create_subset(data_dir, output_path)
