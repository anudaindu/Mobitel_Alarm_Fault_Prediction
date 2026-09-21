import os
import glob
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

def scan_and_create_subset(data_dir: str, output_csv_path: str) -> pd.DataFrame:
    """
    Scans all 36 Huawei telecom alarm Excel files, identifies top 20 outage towers 
    and 15 healthy towers, and extracts raw alarm log rows into a consolidated subset CSV.
    """
    files = sorted(glob.glob(os.path.join(data_dir, 'AlarmLogs*.xlsx')))
    if not files:
        raise FileNotFoundError(f"No AlarmLogs*.xlsx files found in {data_dir}")
        
    print(f"Scanning {len(files)} Huawei Excel log files in {data_dir}...")
    
    dfs = []
    for file_path in files:
        try:
            # Dynamic header row identification
            df_raw = pd.read_excel(file_path, header=None, nrows=12, engine='calamine')
            header_idx = 5
            for idx, row in df_raw.iterrows():
                row_str = " ".join([str(v) for v in row.values if pd.notna(v)])
                if 'Severity' in row_str and ('Alarm ID' in row_str or 'Occurred' in row_str or 'MO Name' in row_str):
                    header_idx = idx
                    break
                    
            df = pd.read_excel(file_path, skiprows=header_idx, engine='calamine')
            df.columns = [str(c).strip() for c in df.columns]
            
            time_col = None
            for candidate in ['Occurred On (NT)', 'Occurred On', 'Event_Time', 'Occurred On (ST)']:
                if candidate in df.columns:
                    time_col = candidate
                    break
                    
            if time_col:
                df.rename(columns={time_col: 'Occurred On (NT)'}, inplace=True)
                df.dropna(subset=['Occurred On (NT)'], inplace=True)
                dfs.append(df)
        except Exception as e:
            print(f"Warning: Failed to load {os.path.basename(file_path)}: {e}")
            
    if not dfs:
        raise ValueError("No valid alarm data could be parsed.")
        
    full_df = pd.concat(dfs, ignore_index=True)
    
    # Standardize Site ID (eNodeB ID if available; else Location Information)
    if 'eNodeB ID' in full_df.columns:
        full_df['eNodeB_clean'] = full_df['eNodeB ID'].astype(str).str.strip()
        full_df['site_id'] = full_df['eNodeB_clean'].apply(lambda x: None if x in ['-', 'nan', 'None', ''] else x)
    else:
        full_df['site_id'] = None

    if 'Location Information' in full_df.columns:
        full_df['site_id'] = full_df['site_id'].fillna(full_df['Location Information'])
    elif 'Alarm Source' in full_df.columns:
        full_df['site_id'] = full_df['site_id'].fillna(full_df['Alarm Source'])
    elif 'MO Name' in full_df.columns:
        full_df['site_id'] = full_df['site_id'].fillna(full_df['MO Name'])

    full_df.dropna(subset=['site_id', 'Occurred On (NT)'], inplace=True)
    full_df['site_id'] = full_df['site_id'].astype(str)
    
    alarm_name_col = 'Name' if 'Name' in full_df.columns else 'Alarm_Name'
    full_df['alarm_name'] = full_df[alarm_name_col].astype(str)
    full_df['is_outage'] = full_df['alarm_name'].str.lower().str.contains('cell unavailable')

    # Select top 20 outage towers & 15 healthy towers
    site_outage_counts = full_df.groupby('site_id')['is_outage'].sum()
    top_20_outage_towers = site_outage_counts.sort_values(ascending=False).head(20).index.tolist()

    healthy_towers_series = site_outage_counts[site_outage_counts == 0]
    healthy_tower_activity = full_df[full_df['site_id'].isin(healthy_towers_series.index)]['site_id'].value_counts()
    healthy_15_towers = healthy_tower_activity.head(15).index.tolist()

    selected_35_towers = top_20_outage_towers + healthy_15_towers
    
    subset_df = full_df[full_df['site_id'].isin(selected_35_towers)].copy()
    subset_df.sort_values(by=['site_id', 'Occurred On (NT)'], inplace=True)
    
    os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)
    subset_df.to_csv(output_csv_path, index=False)
    print(f"Extracted {len(subset_df)} subset rows across {len(selected_35_towers)} towers to {output_csv_path}")
    return subset_df

if __name__ == '__main__':
    data_dir = os.path.expanduser('~/Downloads/AlarmLogs20260803113740030')
    output_path = os.path.expanduser('~/Desktop/mobitel project/Mobitel_Alarm_Fault_Prediction/july_2026_subset_dataset.csv')
    scan_and_create_subset(data_dir, output_path)
