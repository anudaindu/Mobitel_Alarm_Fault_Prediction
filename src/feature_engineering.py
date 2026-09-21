import os
import pandas as pd
import numpy as np

def generate_sliding_window_features(subset_csv_path: str, output_features_path: str) -> pd.DataFrame:
    """
    Computes a 2-hour step sliding window across July 2026 for site-level features:
    - Grouping: Strictly by physical Site ID (site_id)
    - Distinct Identifiers: Preserves enodeb_id (4G) and gnodeb_id (5G) as separate columns.
    - Lookback Window (X): Past 6 hours [T-6h, T) and past 24 hours [T-24h, T)
    - Prediction Window (Y): Next 2 hours [T, T+2h)
    """
    print(f"Reading subset dataset from {subset_csv_path}...")
    df = pd.read_csv(subset_csv_path, low_memory=False)
    
    time_col = 'event_time' if 'event_time' in df.columns else 'Occurred On (NT)'
    df['event_time'] = pd.to_datetime(df[time_col])
    df.sort_values(by=['site_id', 'event_time'], inplace=True)

    if 'enodeb_id' not in df.columns:
        df['enodeb_id'] = '-'
    if 'gnodeb_id' not in df.columns:
        df['gnodeb_id'] = '-'

    df['is_critical'] = (df['severity'].astype(str).str.lower() == 'critical').astype(int)
    df['is_major'] = (df['severity'].astype(str).str.lower() == 'major').astype(int)

    alarm_name_str = df['alarm_name'].astype(str).str.lower()
    rru_pattern = r'rf|rru|vswr'
    bbu_pattern = r'bbu|board|subrack|transmission'
    outage_pattern = r'cell unavailable'

    df['is_rru'] = alarm_name_str.str.contains(rru_pattern, regex=True).astype(int)
    df['is_bbu'] = alarm_name_str.str.contains(bbu_pattern, regex=True).astype(int)
    df['is_outage'] = alarm_name_str.str.contains(outage_pattern, regex=True).astype(int)

    start_time = pd.Timestamp('2026-07-01 00:00:00')
    end_time = pd.Timestamp('2026-07-31 22:00:00')
    time_points = pd.date_range(start=start_time, end=end_time, freq='2h')

    sites = df['site_id'].unique()
    feature_rows = []

    print(f"Generating sliding window features for {len(sites)} physical sites across {len(time_points)} time steps...")

    # Map physical site_id to eNodeB ID and gNodeB ID
    site_enodeb_map = df.groupby('site_id')['enodeb_id'].apply(lambda s: s[s != '-'].iloc[0] if (s != '-').any() else '-').to_dict()
    site_gnodeb_map = df.groupby('site_id')['gnodeb_id'].apply(lambda s: s[s != '-'].iloc[0] if (s != '-').any() else '-').to_dict()

    for site in sites:
        site_df = df[df['site_id'] == site]
        times = site_df['event_time'].values
        is_crit = site_df['is_critical'].values
        is_maj = site_df['is_major'].values
        is_rru = site_df['is_rru'].values
        is_bbu = site_df['is_bbu'].values
        is_out = site_df['is_outage'].values

        enodeb_val = site_enodeb_map.get(site, '-')
        gnodeb_val = site_gnodeb_map.get(site, '-')

        for t in time_points:
            t_np = t.to_datetime64()
            lookback_6h_start = (t - pd.Timedelta(hours=6)).to_datetime64()
            lookback_24h_start = (t - pd.Timedelta(hours=24)).to_datetime64()
            predict_end = (t + pd.Timedelta(hours=2)).to_datetime64()

            idx_6h = (times >= lookback_6h_start) & (times < t_np)
            total_alarms_6h = int(np.sum(idx_6h))
            critical_alarms_6h = int(np.sum(is_crit[idx_6h]))
            major_alarms_6h = int(np.sum(is_maj[idx_6h]))
            rru_alarms_6h = int(np.sum(is_rru[idx_6h]))
            bbu_alarms_6h = int(np.sum(is_bbu[idx_6h]))

            # Streak / Sequence metrics from notebook 04
            crit_streak_6h = 0
            if total_alarms_6h > 1:
                crit_arr = is_crit[idx_6h]
                crit_streak_6h = int(np.max(np.convolve(crit_arr, np.ones(2, dtype=int), mode='valid') == 2)) if len(crit_arr) >= 2 else 0

            idx_24h = (times >= lookback_24h_start) & (times < t_np)
            total_alarms_24h = int(np.sum(idx_24h))

            idx_2h_target = (times >= t_np) & (times < predict_end)
            target_outage_next_2h = 1 if np.sum(is_out[idx_2h_target]) > 0 else 0

            feature_rows.append({
                'site_id': site,
                'enodeb_id': enodeb_val,
                'gnodeb_id': gnodeb_val,
                'window_timestamp': t,
                'total_alarms_6h': total_alarms_6h,
                'critical_alarms_6h': critical_alarms_6h,
                'major_alarms_6h': major_alarms_6h,
                'rru_alarms_6h': rru_alarms_6h,
                'bbu_alarms_6h': bbu_alarms_6h,
                'consecutive_critical_streak_6h': crit_streak_6h,
                'total_alarms_24h': total_alarms_24h,
                'target_outage_next_2h': target_outage_next_2h
            })

    features_df = pd.DataFrame(feature_rows)

    # Multicollinearity Audit (>0.85 correlation threshold per notebook 04)
    num_cols = ['total_alarms_6h', 'critical_alarms_6h', 'major_alarms_6h', 'rru_alarms_6h', 'bbu_alarms_6h', 'consecutive_critical_streak_6h', 'total_alarms_24h']
    num_df = features_df[num_cols]
    corr_matrix = num_df.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_cols = [column for column in upper_tri.columns if any(upper_tri[column] > 0.85)]
    if high_corr_cols:
        print(f"[Feature Selection] Dropping high-correlation redundant features (>0.85): {high_corr_cols}")
        features_df.drop(columns=high_corr_cols, inplace=True)

    os.makedirs(os.path.dirname(os.path.abspath(output_features_path)), exist_ok=True)
    features_df.to_csv(output_features_path, index=False)
    print(f"Generated feature matrix shape: {features_df.shape} -> saved to {output_features_path}")
    return features_df

if __name__ == '__main__':
    subset_csv = os.path.expanduser('~/Desktop/mobitel project/Mobitel_Alarm_Fault_Prediction/july_2026_subset_dataset.csv')
    output_features = os.path.expanduser('~/Desktop/mobitel project/Mobitel_Alarm_Fault_Prediction/telecom_features_july_2026.csv')
    generate_sliding_window_features(subset_csv, output_features)
