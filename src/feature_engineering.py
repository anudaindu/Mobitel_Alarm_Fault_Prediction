import os
import sys
import pandas as pd
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)
from src.config import load_config, get_active_targets, get_active_features, get_lookback_windows

def generate_sliding_window_features(subset_csv_path: str, output_features_path: str, config: dict = None) -> pd.DataFrame:
    """
    Config-Driven Feature Engineering Engine:
    1. Reads runtime settings from config/settings.json.
    2. Dynamically includes/excludes feature categories (BBU, RRU, POWER) based on ACTIVE_FEATURES.
    3. Computes site-level rolling aggregations across active lookback windows (e.g. 6h, 24h, 14d, 30d).
    4. Generates independent binary target columns (target_cell_unavailable_2h, target_cell_outage_2h, etc.).
    5. Dynamically combines active target flags into target_outage_next_2h.
    """
    if config is None:
        config = load_config()

    active_targets = config.get("ACTIVE_TARGETS", [
        "OUTAGE_CELL_UNAVAILABLE", "OUTAGE_CELL_OUTAGE", "OUTAGE_CELL_FAULT", "OUTAGE_SERVICE_UNAVAILABLE"
    ])
    active_features = config.get("ACTIVE_FEATURES", ["BBU_ALARM", "RRU_ALARM", "POWER_ALARM"])
    lookback_windows = config.get("LOOKBACK_WINDOWS_HOURS", [6, 24, 336, 720])

    print(f"Reading subset dataset from {subset_csv_path}...")
    df = pd.read_csv(subset_csv_path, low_memory=False)

    time_col = 'event_time' if 'event_time' in df.columns else 'Occurred On (NT)'
    df['event_time'] = pd.to_datetime(df[time_col])

    # Ensure site_id and node_identifier columns exist
    if 'site_id' not in df.columns:
        df['site_id'] = '-'
    if 'node_identifier' not in df.columns:
        df['node_identifier'] = '-'

    # Primary aggregation entity: site_id if available, otherwise node_identifier
    df['primary_entity'] = np.where(
        (df['site_id'] != '-') & (df['site_id'] != 'UNKNOWN') & (df['site_id'].notna()),
        df['site_id'],
        df['node_identifier']
    )

    df.sort_values(by=['primary_entity', 'event_time'], inplace=True)

    # Identifiers
    df['enodeb_id'] = df['enodeb_id'].astype(str) if 'enodeb_id' in df.columns else '-'
    df['gnodeb_id'] = df['gnodeb_id'].astype(str) if 'gnodeb_id' in df.columns else '-'

    # Alarm Severity
    df['is_critical'] = (df['severity'].astype(str).str.lower() == 'critical').astype(int)
    df['is_major'] = (df['severity'].astype(str).str.lower() == 'major').astype(int)

    # Granular Category Indicators
    name_str = df['alarm_name'].astype(str).str.lower()
    df['is_cell_unavail'] = name_str.str.contains('cell unavailable').astype(int)
    df['is_cell_outage'] = (name_str.str.contains('cell outage') | name_str.str.contains('cell down')).astype(int)
    df['is_cell_fault'] = name_str.str.contains('cell fault').astype(int)
    df['is_service_unavail'] = (name_str.str.contains('service unavailable') | name_str.str.contains('service degradation')).astype(int)

    df['is_rru'] = name_str.str.contains(r'rf|rru|vswr|feeder|antenna', regex=True).astype(int)
    df['is_bbu'] = name_str.str.contains(r'bbu|board|subrack|transmission|baseband', regex=True).astype(int)
    df['is_power'] = name_str.str.contains(r'power|mains|rectifier|battery|ac fail|dc fail', regex=True).astype(int)

    # Define time steps (2h step)
    min_time = df['event_time'].min().floor('2h')
    max_time = df['event_time'].max().ceil('2h')
    time_points = pd.date_range(start=min_time, end=max_time, freq='2h')

    entities = df['primary_entity'].unique()
    feature_rows = []

    print(f"Generating dynamic features for {len(entities)} entities across {len(time_points)} time steps...")
    print(f"Active Features Config: {active_features}")
    print(f"Active Targets Config: {active_targets}")
    print(f"Lookback Windows (Hours): {lookback_windows}")

    # Entity maps
    entity_site_map = df.groupby('primary_entity')['site_id'].first().to_dict()
    entity_node_map = df.groupby('primary_entity')['node_identifier'].first().to_dict()
    entity_enodeb_map = df.groupby('primary_entity')['enodeb_id'].first().to_dict()
    entity_gnodeb_map = df.groupby('primary_entity')['gnodeb_id'].first().to_dict()

    for ent in entities:
        ent_df = df[df['primary_entity'] == ent]
        times = ent_df['event_time'].values

        is_crit = ent_df['is_critical'].values
        is_maj = ent_df['is_major'].values

        is_cell_unavail_arr = ent_df['is_cell_unavail'].values
        is_cell_outage_arr = ent_df['is_cell_outage'].values
        is_cell_fault_arr = ent_df['is_cell_fault'].values
        is_service_unavail_arr = ent_df['is_service_unavail'].values

        is_rru_arr = ent_df['is_rru'].values
        is_bbu_arr = ent_df['is_bbu'].values
        is_power_arr = ent_df['is_power'].values

        site_val = entity_site_map.get(ent, '-')
        node_val = entity_node_map.get(ent, ent)
        enodeb_val = entity_enodeb_map.get(ent, '-')
        gnodeb_val = entity_gnodeb_map.get(ent, '-')

        for t in time_points:
            t_np = t.to_datetime64()
            predict_end = (t + pd.Timedelta(hours=2)).to_datetime64()

            row_data = {
                'site_id': site_val,
                'node_identifier': node_val,
                'enodeb_id': enodeb_val,
                'gnodeb_id': gnodeb_val,
                'window_timestamp': t
            }

            # 1. Rolling Lookback Window Features
            for hours in lookback_windows:
                start_window = (t - pd.Timedelta(hours=hours)).to_datetime64()
                idx_w = (times >= start_window) & (times < t_np)

                total_alarms = int(np.sum(idx_w))
                crit_alarms = int(np.sum(is_crit[idx_w]))
                maj_alarms = int(np.sum(is_maj[idx_w]))

                row_data[f'total_alarms_{hours}h'] = total_alarms
                row_data[f'critical_alarms_{hours}h'] = crit_alarms
                row_data[f'major_alarms_{hours}h'] = maj_alarms

                # Dynamic feature categories based on ACTIVE_FEATURES
                if 'BBU_ALARM' in active_features:
                    row_data[f'bbu_alarms_{hours}h'] = int(np.sum(is_bbu_arr[idx_w]))
                if 'RRU_ALARM' in active_features:
                    row_data[f'rru_alarms_{hours}h'] = int(np.sum(is_rru_arr[idx_w]))
                if 'POWER_ALARM' in active_features:
                    row_data[f'power_alarms_{hours}h'] = int(np.sum(is_power_arr[idx_w]))

            # Consecutive streak metric for primary window (6h or min window)
            primary_h = lookback_windows[0] if lookback_windows else 6
            start_primary = (t - pd.Timedelta(hours=primary_h)).to_datetime64()
            idx_primary = (times >= start_primary) & (times < t_np)
            crit_streak = 0
            if int(np.sum(idx_primary)) > 1:
                crit_arr = is_crit[idx_primary]
                crit_streak = int(np.max(np.convolve(crit_arr, np.ones(2, dtype=int), mode='valid') == 2)) if len(crit_arr) >= 2 else 0
            row_data[f'consecutive_critical_streak_{primary_h}h'] = crit_streak

            # 2. Independent Binary Prediction Targets (Next 2 Hours)
            idx_target = (times >= t_np) & (times < predict_end)

            target_cell_unavail = 1 if np.sum(is_cell_unavail_arr[idx_target]) > 0 else 0
            target_cell_outage = 1 if np.sum(is_cell_outage_arr[idx_target]) > 0 else 0
            target_cell_fault = 1 if np.sum(is_cell_fault_arr[idx_target]) > 0 else 0
            target_service_unavail = 1 if np.sum(is_service_unavail_arr[idx_target]) > 0 else 0

            row_data['target_cell_unavailable_2h'] = target_cell_unavail
            row_data['target_cell_outage_2h'] = target_cell_outage
            row_data['target_cell_fault_2h'] = target_cell_fault
            row_data['target_service_unavailable_2h'] = target_service_unavail

            # Combined target based strictly on ACTIVE_TARGETS
            target_flags = []
            if "OUTAGE_CELL_UNAVAILABLE" in active_targets:
                target_flags.append(target_cell_unavail)
            if "OUTAGE_CELL_OUTAGE" in active_targets:
                target_flags.append(target_cell_outage)
            if "OUTAGE_CELL_FAULT" in active_targets:
                target_flags.append(target_cell_fault)
            if "OUTAGE_SERVICE_UNAVAILABLE" in active_targets:
                target_flags.append(target_service_unavail)

            row_data['target_outage_next_2h'] = 1 if (target_flags and any(tf == 1 for tf in target_flags)) else 0

            feature_rows.append(row_data)

    features_df = pd.DataFrame(feature_rows)

    os.makedirs(os.path.dirname(os.path.abspath(output_features_path)), exist_ok=True)
    features_df.to_csv(output_features_path, index=False)
    print(f"Generated dynamic feature matrix shape: {features_df.shape} -> saved to {output_features_path}")
    return features_df

if __name__ == '__main__':
    subset_csv = os.path.join(BASE_DIR, 'data', 'july_2026_subset_dataset.csv')
    output_features = os.path.join(BASE_DIR, 'data', 'telecom_features_july_2026.csv')
    generate_sliding_window_features(subset_csv, output_features)
