import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)
from src.config import (
    load_config, get_active_targets, get_active_features, get_lookback_windows,
    get_cache_path, get_delta_logs_dir, get_sequence_state_path, get_decay_lambda
)


# ---------------------------------------------------------------------------
# TIME-DECAY WEIGHTING ENGINE
# ---------------------------------------------------------------------------

def apply_time_decay(df: pd.DataFrame, reference_time: datetime = None, lambda_: float = None) -> pd.DataFrame:
    """
    Applies exponential time-decay weights to each row based on how many days
    have elapsed since the event_time relative to reference_time.

    Formula:  w(t) = exp(-lambda_ * delta_days)

    Parameters
    ----------
    df : pd.DataFrame
        Input alarm DataFrame. Must contain an 'event_time' column.
    reference_time : datetime, optional
        Anchor point for decay (default: now).
    lambda_ : float, optional
        Decay rate lambda. If None, reads from config (default 0.05).
        Examples: 1 day → 0.95, 7 days → 0.70, 60 days → 0.05

    Returns
    -------
    pd.DataFrame with an added 'time_decay_weight' column.
    """
    if lambda_ is None:
        lambda_ = get_decay_lambda()
    if reference_time is None:
        reference_time = datetime.now()

    df = df.copy()
    df['event_time'] = pd.to_datetime(df['event_time'])
    delta_days = (reference_time - df['event_time']).dt.total_seconds() / 86400.0
    delta_days = delta_days.clip(lower=0)
    df['time_decay_weight'] = np.exp(-lambda_ * delta_days)
    return df


# ---------------------------------------------------------------------------
# ALARM IMPACT WEIGHT ENGINE
# ---------------------------------------------------------------------------

# Historical impact weights for each alarm category (learned via XGBoost feature importance).
# Override via `alarm_impact_weights` dict in calling code if needed.
DEFAULT_ALARM_IMPACT_WEIGHTS = {
    'is_power':         0.90,   # Mains Input Failure / Battery Voltage Low
    'is_bbu':           0.75,   # BBU / Baseband board faults
    'is_rru':           0.60,   # RF/RRU/VSWR antenna faults
    'is_cell_unavail':  0.85,
    'is_cell_outage':   0.85,
    'is_cell_fault':    0.70,
    'is_service_unavail': 0.65,
    'is_critical':      0.80,
    'is_major':         0.50,
}

def compute_dual_weight(df: pd.DataFrame, alarm_impact_weights: dict = None,
                        reference_time: datetime = None, lambda_: float = None) -> pd.DataFrame:
    """
    Compute Final Alarm Weight = W_impact × w(t) for each row.

    Adds columns:
      - 'time_decay_weight'
      - 'alarm_impact_weight'  (max impact of any flagged alarm category)
      - 'final_alarm_weight'   (product of both)

    Parameters
    ----------
    df : pd.DataFrame
        Alarm-level DataFrame with boolean/int flag columns like is_power, is_bbu, etc.
    alarm_impact_weights : dict, optional
        Mapping of flag column → float weight. Defaults to DEFAULT_ALARM_IMPACT_WEIGHTS.
    reference_time : datetime, optional
        Anchor time for decay (default: now).
    lambda_ : float, optional
        Decay lambda (default: from config, 0.05).

    Returns
    -------
    pd.DataFrame with 'time_decay_weight', 'alarm_impact_weight', 'final_alarm_weight'.
    """
    if alarm_impact_weights is None:
        alarm_impact_weights = DEFAULT_ALARM_IMPACT_WEIGHTS

    df = apply_time_decay(df, reference_time=reference_time, lambda_=lambda_)

    # Compute alarm impact: max weight of all flagged categories in this row
    impact_values = np.zeros(len(df))
    for col, weight in alarm_impact_weights.items():
        if col in df.columns:
            flagged = (df[col].fillna(0).astype(int) == 1).values
            impact_values = np.where(flagged, np.maximum(impact_values, weight), impact_values)

    df['alarm_impact_weight'] = impact_values
    df['final_alarm_weight'] = df['alarm_impact_weight'] * df['time_decay_weight']
    return df


# ---------------------------------------------------------------------------
# PERSISTENT PREDICTION CACHE LOADER
# ---------------------------------------------------------------------------

def load_prediction_cache(cache_path: str = None) -> pd.DataFrame:
    """
    Loads the persistent prediction state cache from disk.

    Parameters
    ----------
    cache_path : str, optional
        Path to latest_predictions.csv. Defaults to config CACHE_PATH.

    Returns
    -------
    pd.DataFrame of cached predictions, or empty DataFrame if cache doesn't exist.
    """
    if cache_path is None:
        cache_path = get_cache_path()

    if not os.path.exists(cache_path):
        print(f"[Cache] No cache found at {cache_path}. Starting fresh.")
        return pd.DataFrame()

    try:
        df = pd.read_csv(cache_path, low_memory=False)
        if 'window_timestamp' in df.columns:
            df['window_timestamp'] = pd.to_datetime(df['window_timestamp'])
        print(f"[Cache] Loaded {len(df)} cached prediction rows from {cache_path}")
        return df
    except Exception as e:
        print(f"[Cache] Error reading cache {cache_path}: {e}. Returning empty DataFrame.")
        return pd.DataFrame()


def save_prediction_cache(df: pd.DataFrame, cache_path: str = None) -> bool:
    """
    Persists the current prediction state to disk.

    Parameters
    ----------
    df : pd.DataFrame
        Prediction results to persist.
    cache_path : str, optional
        Path to write. Defaults to config CACHE_PATH.

    Returns
    -------
    bool : True on success, False on failure.
    """
    if cache_path is None:
        cache_path = get_cache_path()

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    try:
        df.to_csv(cache_path, index=False)
        print(f"[Cache] Saved {len(df)} rows to {cache_path}")
        return True
    except Exception as e:
        print(f"[Cache] Error writing cache {cache_path}: {e}")
        return False


# ---------------------------------------------------------------------------
# PRECURSOR SEQUENCE STATE PERSISTENCE
# ---------------------------------------------------------------------------

def load_sequence_state(state_path: str = None) -> dict:
    """
    Loads the precursor alarm sequence state from a JSON file.
    Maintains entity-level alarm chain continuity across daily batch runs.

    Returns
    -------
    dict: {entity_id: [list of recent alarm names with timestamps]}
    """
    if state_path is None:
        state_path = get_sequence_state_path()

    if not os.path.exists(state_path):
        print(f"[SequenceState] No state file found at {state_path}. Starting fresh.")
        return {}

    try:
        with open(state_path, 'r') as f:
            state = json.load(f)
        print(f"[SequenceState] Loaded sequence state for {len(state)} entities from {state_path}")
        return state
    except Exception as e:
        print(f"[SequenceState] Error reading state {state_path}: {e}. Starting fresh.")
        return {}


def save_sequence_state(state: dict, state_path: str = None) -> bool:
    """
    Saves the current precursor alarm sequence state to disk for continuity
    across daily batch executions.

    Parameters
    ----------
    state : dict
        Mapping of entity_id to list of recent alarm events (dicts with alarm_name, event_time).
    state_path : str, optional
        Output path. Defaults to config SEQUENCE_STATE_PATH.

    Returns
    -------
    bool : True on success.
    """
    if state_path is None:
        state_path = get_sequence_state_path()

    os.makedirs(os.path.dirname(state_path), exist_ok=True)
    try:
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        print(f"[SequenceState] Saved sequence state for {len(state)} entities to {state_path}")
        return True
    except Exception as e:
        print(f"[SequenceState] Error writing state {state_path}: {e}")
        return False


# ---------------------------------------------------------------------------
# DELTA MERGE ENGINE
# ---------------------------------------------------------------------------

def merge_delta_with_cache(delta_df: pd.DataFrame, cache_df: pd.DataFrame,
                           lookback_hours: int = 720) -> pd.DataFrame:
    """
    Merges the new daily delta alarm log with yesterday's persistent feature cache.

    Strategy:
    - Concatenates delta_df (new raw alarms) with cached alarm-level data
      (if cache contains a 'raw_alarm_data' serialized column — otherwise falls back
       to using only the delta for the current run while preserving the prediction cache).
    - Trims combined log to the lookback_hours window to avoid unbounded growth.
    - Deduplicates on (primary_entity, event_time, alarm_name) to prevent double-counting.

    Parameters
    ----------
    delta_df : pd.DataFrame
        Newly uploaded alarm log for today's run.
    cache_df : pd.DataFrame
        Cached prediction DataFrame from previous run (latest_predictions.csv).
    lookback_hours : int
        How many hours of alarm history to retain (default: 720 = 30 days).

    Returns
    -------
    pd.DataFrame: Merged and deduped alarm-level DataFrame ready for feature generation.
    """
    if delta_df.empty:
        print("[DeltaMerge] No new data in delta. Using cache only.")
        return cache_df

    # Normalise event_time
    if 'event_time' in delta_df.columns:
        delta_df['event_time'] = pd.to_datetime(delta_df['event_time'])
    elif 'Occurred On (NT)' in delta_df.columns:
        delta_df['event_time'] = pd.to_datetime(delta_df['Occurred On (NT)'])

    cutoff = pd.Timestamp.now() - pd.Timedelta(hours=lookback_hours)

    # If cache contains raw alarm rows (identified by alarm_name column)
    if not cache_df.empty and 'alarm_name' in cache_df.columns:
        if 'event_time' in cache_df.columns:
            cache_df['event_time'] = pd.to_datetime(cache_df['event_time'])
        # Keep only rows within lookback window
        recent_cache = cache_df[cache_df['event_time'] >= cutoff]
        merged = pd.concat([recent_cache, delta_df], ignore_index=True)
    else:
        merged = delta_df[delta_df.get('event_time', pd.Series(dtype='datetime64[ns]')) >= cutoff] if 'event_time' in delta_df.columns else delta_df.copy()

    # Deduplicate
    dedup_cols = [c for c in ['primary_entity', 'site_id', 'node_identifier', 'event_time', 'alarm_name'] if c in merged.columns]
    if dedup_cols:
        before = len(merged)
        merged = merged.drop_duplicates(subset=dedup_cols)
        print(f"[DeltaMerge] Deduplicated {before - len(merged)} rows → {len(merged)} rows remain.")

    merged.sort_values(by='event_time', inplace=True)
    print(f"[DeltaMerge] Final merged alarm log: {len(merged)} rows")
    return merged


def load_delta_logs(delta_dir: str = None) -> pd.DataFrame:
    """
    Reads all new CSV files from the daily delta upload directory and concatenates them.

    Parameters
    ----------
    delta_dir : str, optional
        Path to the directory containing daily upload CSVs. Defaults to config DELTA_LOGS_DIR.

    Returns
    -------
    pd.DataFrame: Combined delta DataFrame, empty if no files found.
    """
    if delta_dir is None:
        delta_dir = get_delta_logs_dir()

    os.makedirs(delta_dir, exist_ok=True)

    csv_files = [f for f in os.listdir(delta_dir) if f.lower().endswith('.csv')]
    if not csv_files:
        print(f"[DeltaLoad] No CSV files found in {delta_dir}.")
        return pd.DataFrame()

    frames = []
    for fname in sorted(csv_files):
        fpath = os.path.join(delta_dir, fname)
        try:
            df = pd.read_csv(fpath, low_memory=False)
            df['_source_file'] = fname
            frames.append(df)
            print(f"[DeltaLoad] Loaded {len(df)} rows from {fname}")
        except Exception as e:
            print(f"[DeltaLoad] Failed to read {fname}: {e}")

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    print(f"[DeltaLoad] Combined delta: {len(combined)} rows from {len(frames)} files.")
    return combined


# ---------------------------------------------------------------------------
# CORE FEATURE GENERATION ENGINE (supports both full & incremental modes)
# ---------------------------------------------------------------------------

def generate_sliding_window_features(
    subset_csv_path: str = None,
    output_features_path: str = None,
    config: dict = None,
    input_df: pd.DataFrame = None,
    incremental: bool = False
) -> pd.DataFrame:
    """
    Config-Driven Feature Engineering Engine.

    Supports two execution modes:
    - **Full mode** (incremental=False): Reads from subset_csv_path, rebuilds full matrix.
    - **Incremental mode** (incremental=True): Accepts a pre-merged input_df (delta + cache),
      generates features only for the newest time steps, then appends to existing cache.

    Steps:
    1. Reads runtime settings from config/settings.json.
    2. Dynamically includes/excludes feature categories based on ACTIVE_FEATURES.
    3. Computes site-level rolling aggregations across active lookback windows (e.g. 6h, 24h, 14d, 30d).
    4. Generates independent binary target columns (target_cell_unavailable_2h, etc.).
    5. Dynamically combines active target flags into target_outage_next_2h.

    Parameters
    ----------
    subset_csv_path : str, optional
        Path to full raw alarm CSV (full mode only).
    output_features_path : str, optional
        Path to write feature matrix CSV. If None, skips disk write.
    config : dict, optional
        Pre-loaded config dict. Loaded automatically if None.
    input_df : pd.DataFrame, optional
        Pre-merged alarm DataFrame (incremental mode). Overrides subset_csv_path.
    incremental : bool
        If True, only generates rows for timestamps newer than the cached max timestamp.

    Returns
    -------
    pd.DataFrame: Generated feature matrix.
    """
    if config is None:
        config = load_config()

    active_targets = config.get("ACTIVE_TARGETS", [
        "OUTAGE_CELL_UNAVAILABLE", "OUTAGE_CELL_OUTAGE", "OUTAGE_CELL_FAULT", "OUTAGE_SERVICE_UNAVAILABLE"
    ])
    active_features = config.get("ACTIVE_FEATURES", ["BBU_ALARM", "RRU_ALARM", "POWER_ALARM"])
    lookback_windows = config.get("LOOKBACK_WINDOWS_HOURS", [6, 24, 336, 720])

    # --- Data Loading ---
    if input_df is not None:
        df = input_df.copy()
        print(f"[FeatureEng] Using pre-loaded input DataFrame ({len(df)} rows).")
    elif subset_csv_path is not None:
        print(f"[FeatureEng] Reading subset dataset from {subset_csv_path}...")
        df = pd.read_csv(subset_csv_path, low_memory=False)
    else:
        raise ValueError("Either subset_csv_path or input_df must be provided.")

    time_col = 'event_time' if 'event_time' in df.columns else 'Occurred On (NT)'
    df['event_time'] = pd.to_datetime(df[time_col])

    # Ensure site_id and node_identifier columns exist
    if 'site_id' not in df.columns:
        df['site_id'] = '-'
    if 'node_identifier' not in df.columns:
        df['node_identifier'] = '-'

    # Primary aggregation entity
    df['primary_entity'] = np.where(
        (df['site_id'] != '-') & (df['site_id'] != 'UNKNOWN') & (df['site_id'].notna()),
        df['site_id'],
        df['node_identifier']
    )

    df.sort_values(by=['primary_entity', 'event_time'], inplace=True)

    df['enodeb_id'] = df['enodeb_id'].astype(str) if 'enodeb_id' in df.columns else '-'
    df['gnodeb_id'] = df['gnodeb_id'].astype(str) if 'gnodeb_id' in df.columns else '-'

    # Alarm Severity flags
    df['is_critical'] = (df['severity'].astype(str).str.lower() == 'critical').astype(int)
    df['is_major'] = (df['severity'].astype(str).str.lower() == 'major').astype(int)

    # Granular category indicators
    name_str = df['alarm_name'].astype(str).str.lower()
    df['is_cell_unavail'] = name_str.str.contains('cell unavailable').astype(int)
    df['is_cell_outage'] = (name_str.str.contains('cell outage') | name_str.str.contains('cell down')).astype(int)
    df['is_cell_fault'] = name_str.str.contains('cell fault').astype(int)
    df['is_service_unavail'] = (name_str.str.contains('service unavailable') | name_str.str.contains('service degradation')).astype(int)
    df['is_rru'] = name_str.str.contains(r'rf|rru|vswr|feeder|antenna', regex=True).astype(int)
    df['is_bbu'] = name_str.str.contains(r'bbu|board|subrack|transmission|baseband', regex=True).astype(int)
    df['is_power'] = name_str.str.contains(r'power|mains|rectifier|battery|ac fail|dc fail', regex=True).astype(int)

    # --- Apply Dual-Weighting ---
    df = compute_dual_weight(df, reference_time=datetime.now())

    # --- Define time steps (2h granularity) ---
    min_time = df['event_time'].min().floor('2h')
    max_time = df['event_time'].max().ceil('2h')

    # Incremental mode: only compute time points newer than cached max
    if incremental:
        cache_df = load_prediction_cache()
        if not cache_df.empty and 'window_timestamp' in cache_df.columns:
            cache_max_time = pd.to_datetime(cache_df['window_timestamp']).max()
            min_time = max(min_time, cache_max_time + pd.Timedelta(hours=2))
            print(f"[FeatureEng] Incremental mode: generating from {min_time} onwards.")
        else:
            print("[FeatureEng] Incremental mode: no prior cache, generating full range.")

    time_points = pd.date_range(start=min_time, end=max_time, freq='2h')
    if len(time_points) == 0:
        print("[FeatureEng] No new time steps to compute. Cache is up-to-date.")
        return pd.DataFrame()

    entities = df['primary_entity'].unique()
    feature_rows = []

    print(f"[FeatureEng] Generating features for {len(entities)} entities × {len(time_points)} time steps...")
    print(f"  Active Features: {active_features}")
    print(f"  Active Targets : {active_targets}")
    print(f"  Lookback Hours : {lookback_windows}")

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
        decay_weights = ent_df['time_decay_weight'].values
        alarm_impact = ent_df['alarm_impact_weight'].values

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

            # Rolling lookback window features
            for hours in lookback_windows:
                start_window = (t - pd.Timedelta(hours=hours)).to_datetime64()
                idx_w = (times >= start_window) & (times < t_np)

                # Decay-weighted alarm counts
                w = decay_weights[idx_w]
                total_alarms = int(np.sum(idx_w))
                crit_alarms = int(np.sum(is_crit[idx_w]))
                maj_alarms = int(np.sum(is_maj[idx_w]))
                weighted_alarm_score = float(np.sum(alarm_impact[idx_w] * w))

                row_data[f'total_alarms_{hours}h'] = total_alarms
                row_data[f'critical_alarms_{hours}h'] = crit_alarms
                row_data[f'major_alarms_{hours}h'] = maj_alarms
                row_data[f'weighted_alarm_score_{hours}h'] = round(weighted_alarm_score, 4)

                if 'BBU_ALARM' in active_features:
                    row_data[f'bbu_alarms_{hours}h'] = int(np.sum(is_bbu_arr[idx_w]))
                if 'RRU_ALARM' in active_features:
                    row_data[f'rru_alarms_{hours}h'] = int(np.sum(is_rru_arr[idx_w]))
                if 'POWER_ALARM' in active_features:
                    row_data[f'power_alarms_{hours}h'] = int(np.sum(is_power_arr[idx_w]))

            # Consecutive critical streak in primary window
            primary_h = lookback_windows[0] if lookback_windows else 6
            start_primary = (t - pd.Timedelta(hours=primary_h)).to_datetime64()
            idx_primary = (times >= start_primary) & (times < t_np)
            crit_streak = 0
            if int(np.sum(idx_primary)) > 1:
                crit_arr = is_crit[idx_primary]
                crit_streak = int(np.max(np.convolve(crit_arr, np.ones(2, dtype=int), mode='valid') == 2)) if len(crit_arr) >= 2 else 0
            row_data[f'consecutive_critical_streak_{primary_h}h'] = crit_streak

            # Binary prediction targets (next 2 hours)
            idx_target = (times >= t_np) & (times < predict_end)
            target_cell_unavail = 1 if np.sum(is_cell_unavail_arr[idx_target]) > 0 else 0
            target_cell_outage = 1 if np.sum(is_cell_outage_arr[idx_target]) > 0 else 0
            target_cell_fault = 1 if np.sum(is_cell_fault_arr[idx_target]) > 0 else 0
            target_service_unavail = 1 if np.sum(is_service_unavail_arr[idx_target]) > 0 else 0

            row_data['target_cell_unavailable_2h'] = target_cell_unavail
            row_data['target_cell_outage_2h'] = target_cell_outage
            row_data['target_cell_fault_2h'] = target_cell_fault
            row_data['target_service_unavailable_2h'] = target_service_unavail

            # Combined active target flag
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

    if output_features_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_features_path)), exist_ok=True)
        features_df.to_csv(output_features_path, index=False)
        print(f"[FeatureEng] Feature matrix: {features_df.shape} → saved to {output_features_path}")

    return features_df


if __name__ == '__main__':
    subset_csv = os.path.join(BASE_DIR, 'data', 'july_2026_subset_dataset.csv')
    output_features = os.path.join(BASE_DIR, 'data', 'telecom_features_july_2026.csv')
    generate_sliding_window_features(subset_csv_path=subset_csv, output_features_path=output_features)
