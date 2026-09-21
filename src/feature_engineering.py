import os
import pandas as pd
import numpy as np

def generate_sliding_window_features(subset_csv_path: str, output_features_path: str) -> pd.DataFrame:
    """
    Computes a 2-hour step sliding window across July 2026 for site-level features:
    - Lookback Window (X): Past 6 hours [T-6h, T)
    - Prediction Window (Y): Next 2 hours [T, T+2h)
    
    Features engineered per window:
    - total_alarms_6h
    - critical_alarms_6h
    - major_alarms_6h
    - rru_alarms_6h
    - bbu_alarms_6h
    - target_outage_next_2h (Binary Target)
    """
    df = pd.read_csv(subset_csv_path, low_memory=False)
    df['event_time'] = pd.to_datetime(df['Occurred On (NT)'])
    df.sort_values(by=['site_id', 'event_time'], inplace=True)

    df['is_critical'] = (df['Severity'].astype(str).str.lower() == 'critical').astype(int)
    df['is_major'] = (df['Severity'].astype(str).str.lower() == 'major').astype(int)

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

    for site in sites:
        site_df = df[df['site_id'] == site]
        times = site_df['event_time'].values
        is_crit = site_df['is_critical'].values
        is_maj = site_df['is_major'].values
        is_rru = site_df['is_rru'].values
        is_bbu = site_df['is_bbu'].values
        is_out = site_df['is_outage'].values

        for t in time_points:
            t_np = t.to_datetime64()
            lookback_start = (t - pd.Timedelta(hours=6)).to_datetime64()
            predict_end = (t + pd.Timedelta(hours=2)).to_datetime64()

            # Lookback window [T-6h, T)
            idx_6h = (times >= lookback_start) & (times < t_np)
            total_alarms_6h = int(np.sum(idx_6h))
            critical_alarms_6h = int(np.sum(is_crit[idx_6h]))
            major_alarms_6h = int(np.sum(is_maj[idx_6h]))
            rru_alarms_6h = int(np.sum(is_rru[idx_6h]))
            bbu_alarms_6h = int(np.sum(is_bbu[idx_6h]))

            # Prediction window [T, T+2h)
            idx_2h_target = (times >= t_np) & (times < predict_end)
            target_outage_next_2h = 1 if np.sum(is_out[idx_2h_target]) > 0 else 0

            feature_rows.append({
                'site_id': site,
                'window_timestamp': t,
                'total_alarms_6h': total_alarms_6h,
                'critical_alarms_6h': critical_alarms_6h,
                'major_alarms_6h': major_alarms_6h,
                'rru_alarms_6h': rru_alarms_6h,
                'bbu_alarms_6h': bbu_alarms_6h,
                'target_outage_next_2h': target_outage_next_2h
            })

    features_df = pd.DataFrame(feature_rows)
    os.makedirs(os.path.dirname(os.path.abspath(output_features_path)), exist_ok=True)
    features_df.to_csv(output_features_path, index=False)
    print(f"Generated feature matrix shape: {features_df.shape} -> saved to {output_features_path}")
    return features_df

if __name__ == '__main__':
    subset_csv = os.path.expanduser('~/Desktop/mobitel project/Mobitel_Alarm_Fault_Prediction/july_2026_subset_dataset.csv')
    output_features = os.path.expanduser('~/Desktop/mobitel project/Mobitel_Alarm_Fault_Prediction/telecom_features_july_2026.csv')
    generate_sliding_window_features(subset_csv, output_features)
