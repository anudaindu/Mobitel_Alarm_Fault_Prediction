import os
import re
import glob
import pandas as pd
import numpy as np

PARQUET_CACHE = os.path.expanduser('~/.gemini/antigravity-ide/brain/12b58479-4547-4151-84bf-3e80a1ebd527/scratch/combined_raw_alarms.parquet')

# Strict 6-Character Site ID Regex: 1 Letter + 4 Alphanumeric + 1 Digit
SITE_ID_REGEX = re.compile(r'^[A-Za-z][A-Za-z0-9]{4}[0-9]$')

# Explicit Enumerated Alarm Lists
TARGET_ALARM_MAPPINGS = {
    'is_cell_unavailable': ["cell unavailable", "lte cell unavailable", "nr cell unavailable"],
    'is_cell_outage': ["cell outage", "cell service outage", "cell down"],
    'is_cell_fault': ["cell fault", "cell hardware fault"],
    'is_service_unavailable': ["service unavailable", "rf service unavailable", "service degradation"]
}

FEATURE_ALARM_MAPPINGS = {
    'is_bbu': [
        "bbu board fault", "bbu cpri interface error", "bbu optical module fault",
        "bbu maintenance subrack fault", "bbu slot abnormal", "main control board offline",
        "bbu clock subsystem unsynchronized", "bbu power board fault", "bbu", "board", "subrack", "transmission", "baseband"
    ],
    'is_rru': [
        "rru rf unit maintenance fault", "rf unit vswr threshold exceeded", "rru transmit power abnormal",
        "antenna line device fault", "rru cpri interface error", "rf unit gain imbalance",
        "feeder line outage", "rru high temperature", "rf", "rru", "vswr", "feeder", "antenna"
    ],
    'is_power': [
        "mains input failure", "rectifier module failure", "battery voltage low",
        "ac power off", "dc high/low voltage", "generator failure",
        "power subsystem temperature high", "mains supply abnormal", "power", "mains", "rectifier", "battery", "ac fail", "dc fail"
    ]
}

def standardize_column_name(col_name: str) -> str:
    """Standardizes column header formatting."""
    col_name = str(col_name).strip().lower()
    col_name = col_name.replace(' ', '_')
    col_name = re.sub(r'[^a-z0-9_]', '', col_name)
    return col_name

def parse_duration_minutes(duration_str) -> float:
    """Parses duration string into numeric minutes."""
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

def extract_strict_site_id(row: pd.Series) -> tuple:
    """
    Extracts strict 6-character Site ID using fallback candidate order:
    1. Alarm Source -> 2. MO Name -> 3. BBU Name -> 4. NE Name -> 5. Location Information
    
    Rule:
    - Candidate prefix before first '_' must match ^[A-Za-z][A-Za-z0-9]{4}[0-9]$
    - Valid examples: ZBIY43, GLIND1
    - If invalid or missing, sets site_id = '-' and keeps raw string in node_identifier.
    """
    candidate_cols = ['alarm_source', 'mo_name', 'bbu_name', 'ne_name', 'location_information', 'enodeb_id', 'gnodeb_id']
    raw_identifier = "-"

    for col in candidate_cols:
        if col in row and pd.notna(row[col]):
            val_str = str(row[col]).strip()
            if not val_str or val_str in ['-', 'nan', 'none', 'UNKNOWN', 'OSS'] or val_str.startswith('UGW'):
                continue

            if raw_identifier == "-":
                raw_identifier = val_str

            # Strip prefixes like eNodeB Function Name= or NE Name=
            clean_str = val_str.split('=')[-1].strip() if '=' in val_str else val_str
            tokens = clean_str.split('_')
            prefix = tokens[0].strip()

            if SITE_ID_REGEX.match(prefix):
                return prefix.upper(), clean_str

    return "-", raw_identifier

def categorize_alarm_event(alarm_name: str) -> dict:
    """
    Categorizes raw alarm strings strictly using explicit enumerated alarm lists.
    """
    name_clean = str(alarm_name).lower().strip()
    result = {}

    for flag_col, enumerated_list in TARGET_ALARM_MAPPINGS.items():
        result[flag_col] = 1 if any(kw in name_clean for kw in enumerated_list) else 0

    for flag_col, enumerated_list in FEATURE_ALARM_MAPPINGS.items():
        result[flag_col] = 1 if any(kw in name_clean for kw in enumerated_list) else 0

    return result

def clean_raw_alarm_dataframe(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Applies integrated notebook cleaning, strict 6-char site ID parsing & enumerated alarm mapping:
    1. Standardizes column headers.
    2. Parses strict 6-character site_id (^[A-Za-z][A-Za-z0-9]{4}[0-9]$) or sets site_id='-'.
    3. Preserves full raw node_identifier, enodeb_id (4G), gnodeb_id (5G) as separate distinct columns.
    4. Categorizes alarms into target & feature flags using explicit enumerated lists.
    5. Standardizes datetime & deduplicates repeat logs.
    """
    df = df_raw.copy()
    df.columns = [standardize_column_name(c) for c in df.columns]

    time_col = None
    for candidate in ['occurred_on_nt', 'occurred_on', 'event_time', 'occurred_on_st']:
        if candidate in df.columns:
            time_col = candidate
            break

    if not time_col:
        raise KeyError(f"No valid event time column found in DataFrame columns: {list(df.columns)}")

    df['event_time'] = pd.to_datetime(df[time_col], errors='coerce')
    df.dropna(subset=['event_time'], inplace=True)

    df['enodeb_id'] = df['enodeb_id'].astype(str).str.strip().apply(lambda x: '-' if x in ['-', 'nan', 'none', ''] else x) if 'enodeb_id' in df.columns else '-'
    df['gnodeb_id'] = df['gnodeb_id'].astype(str).str.strip().apply(lambda x: '-' if x in ['-', 'nan', 'none', ''] else x) if 'gnodeb_id' in df.columns else '-'

    # Strict 6-Character Site ID & Node Identifier Extraction
    site_tuples = df.apply(extract_strict_site_id, axis=1)
    df['site_id'] = [t[0] for t in site_tuples]
    df['node_identifier'] = [t[1] for t in site_tuples]

    df['node_identifier'] = np.where(
        df['node_identifier'] == '-',
        np.where(df['enodeb_id'] != '-', df['enodeb_id'], df['gnodeb_id']),
        df['node_identifier']
    )

    alarm_name_col = 'name' if 'name' in df.columns else ('alarm_name' if 'alarm_name' in df.columns else None)
    df['alarm_name'] = df[alarm_name_col].astype(str) if alarm_name_col else 'Unknown Alarm'
    df['severity'] = df['severity'].astype(str).str.capitalize() if 'severity' in df.columns else 'Major'
    df['duration_minutes'] = df['alarm_duration'].apply(parse_duration_minutes) if 'alarm_duration' in df.columns else 0.0

    # Categorize alarms using explicit enumerated list matching
    cats = df['alarm_name'].apply(categorize_alarm_event).apply(pd.Series)
    df = pd.concat([df, cats], axis=1)

    df['is_outage'] = df['is_cell_unavailable']

    df['event_minute'] = df['event_time'].dt.floor('min')
    df.drop_duplicates(subset=['site_id', 'node_identifier', 'event_minute', 'alarm_name', 'severity'], inplace=True)
    df.drop(columns=['event_minute'], inplace=True, errors='ignore')

    df.sort_values(by=['event_time'], inplace=True)
    return df

def scan_and_create_subset(data_dir: str, output_csv_path: str) -> pd.DataFrame:
    """Scans raw Huawei log files, applies strict 6-character Site ID regex parsing, and exports subset dataset."""
    if os.path.exists(PARQUET_CACHE):
        print(f"Loading raw alarms from cache {PARQUET_CACHE}...")
        full_raw_df = pd.read_parquet(PARQUET_CACHE)
        full_df = clean_raw_alarm_dataframe(full_raw_df)
    else:
        files = sorted(glob.glob(os.path.join(data_dir, 'AlarmLogs*.xlsx')))
        if not files:
            raise FileNotFoundError(f"No AlarmLogs*.xlsx files found in {data_dir}")

        print(f"Scanning {len(files)} Huawei log files with Strict 6-Character Site ID parsing...")
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

    full_df['group_id'] = np.where(full_df['site_id'] != '-', full_df['site_id'], full_df['node_identifier'])

    site_outage_counts = full_df.groupby('group_id')['is_cell_unavailable'].sum()
    top_20_outage_towers = site_outage_counts.sort_values(ascending=False).head(20).index.tolist()

    healthy_towers_series = site_outage_counts[site_outage_counts == 0]
    healthy_tower_activity = full_df[full_df['group_id'].isin(healthy_towers_series.index)]['group_id'].value_counts()
    healthy_15_towers = healthy_tower_activity.head(15).index.tolist()

    selected_35_towers = top_20_outage_towers + healthy_15_towers
    subset_df = full_df[full_df['group_id'].isin(selected_35_towers)].copy()
    subset_df.drop(columns=['group_id'], inplace=True, errors='ignore')
    subset_df.sort_values(by=['event_time'], inplace=True)

    os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)
    subset_df.to_csv(output_csv_path, index=False)
    print(f"Cleaned subset exported: {len(subset_df)} rows across {len(selected_35_towers)} towers -> {output_csv_path}")
    return subset_df

if __name__ == '__main__':
    data_dir = os.path.expanduser('~/Downloads/AlarmLogs20260803113740030')
    output_path = os.path.expanduser('~/Desktop/mobitel project/Mobitel_Alarm_Fault_Prediction/july_2026_subset_dataset.csv')
    scan_and_create_subset(data_dir, output_path)
