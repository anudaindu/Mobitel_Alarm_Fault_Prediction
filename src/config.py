import os
import json

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CONFIG_DIR = os.path.join(BASE_DIR, 'config')
SETTINGS_FILE = os.path.join(CONFIG_DIR, 'settings.json')

DEFAULT_CONFIG = {
    "ACTIVE_TARGETS": [
        "OUTAGE_CELL_UNAVAILABLE",
        "OUTAGE_CELL_OUTAGE",
        "OUTAGE_CELL_FAULT",
        "OUTAGE_SERVICE_UNAVAILABLE"
    ],
    "ACTIVE_FEATURES": [
        "BBU_ALARM",
        "RRU_ALARM",
        "POWER_ALARM"
    ],
    "LOOKBACK_WINDOWS_HOURS": [
        6,
        24,
        336,
        720
    ],
    "ENVIRONMENT_MODE": "STAGING",
    "DECAY_LAMBDA": 0.05,
    "CACHE_PATH": "data/latest_predictions.csv",
    "DELTA_LOGS_DIR": "data/daily_uploads",
    "SEQUENCE_STATE_PATH": "data/precursor_state.json"
}

def load_config() -> dict:
    """Loads runtime settings from config/settings.json with default fallbacks."""
    if not os.path.exists(SETTINGS_FILE):
        save_config(DEFAULT_CONFIG)
        return DEFAULT_CONFIG.copy()
    try:
        with open(SETTINGS_FILE, 'r') as f:
            data = json.load(f)
            # Ensure all keys exist
            config = DEFAULT_CONFIG.copy()
            config.update(data)
            return config
    except Exception as e:
        print(f"Error loading {SETTINGS_FILE}: {e}. Returning default config.")
        return DEFAULT_CONFIG.copy()

def save_config(config_dict: dict) -> bool:
    """Saves runtime settings directly to config/settings.json."""
    os.makedirs(CONFIG_DIR, exist_ok=True)
    try:
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(config_dict, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving {SETTINGS_FILE}: {e}")
        return False

def get_active_targets() -> list:
    return load_config().get("ACTIVE_TARGETS", DEFAULT_CONFIG["ACTIVE_TARGETS"])

def get_active_features() -> list:
    return load_config().get("ACTIVE_FEATURES", DEFAULT_CONFIG["ACTIVE_FEATURES"])

def get_lookback_windows() -> list:
    return load_config().get("LOOKBACK_WINDOWS_HOURS", DEFAULT_CONFIG["LOOKBACK_WINDOWS_HOURS"])

def get_environment_mode() -> str:
    return load_config().get("ENVIRONMENT_MODE", DEFAULT_CONFIG["ENVIRONMENT_MODE"])

def get_cache_path() -> str:
    """Return the absolute path to the prediction cache CSV."""
    return os.path.join(BASE_DIR, load_config().get("CACHE_PATH", "data/latest_predictions.csv"))

def get_delta_logs_dir() -> str:
    """Return the absolute path to the directory containing daily delta log CSVs."""
    return os.path.join(BASE_DIR, load_config().get("DELTA_LOGS_DIR", "data/daily_uploads"))

def get_sequence_state_path() -> str:
    """Return the absolute path to the precursor sequence state JSON file."""
    return os.path.join(BASE_DIR, load_config().get("SEQUENCE_STATE_PATH", "data/precursor_state.json"))

def get_decay_lambda() -> float:
    """Return the decay lambda value for time decay weighting."""
    return float(load_config().get("DECAY_LAMBDA", 0.05))

if __name__ == '__main__':
    cfg = load_config()
    print("Loaded configuration successfully:", cfg)
