import os
import json

CONFIG_DIR = os.path.expanduser('~/Desktop/mobitel project/Mobitel_Alarm_Fault_Prediction/config')
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
    "ENVIRONMENT_MODE": "STAGING"
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

if __name__ == '__main__':
    cfg = load_config()
    print("Loaded configuration successfully:", cfg)
