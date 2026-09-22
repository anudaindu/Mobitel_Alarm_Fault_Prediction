# 4G/5G Telecom Tower Outage Prediction System

[![Site Launch Status](https://img.shields.io/badge/Live_Site-ONLINE-06D6A0?style=for-the-badge)](http://localhost:8502)
[![Deployment Mode](https://img.shields.io/badge/Deployment_Mode-STAGING_|_PRODUCTION-1E3A8A?style=for-the-badge)](http://localhost:8502)

An end-to-end autonomous data engineering, exploratory data analysis (EDA), and machine learning framework for Mobitel's 4G/5G infrastructure. Features a **Dynamic Configuration Engine**, **Password-Protected Admin Panel**, **XGBoost probability calibration**, and an **Airflow DAG with a 60-day auto-purge policy**.

---

## 🚀 Site Launch Link

The interactive Streamlit Web Application is launched and running locally:

- **Local Launch Link**: [http://localhost:8502](http://localhost:8502)
- **NOC Admin Password**: `mobitel123` (managed via `ADMIN_PASSWORD` environment variable)

---

## 🛠️ Key Platform Features

1. **Dynamic Configuration Engine (`config/settings.json`)**:
   - Dynamically toggles target labels (`ACTIVE_TARGETS`), feature categories (`ACTIVE_FEATURES`), lookback horizons (`LOOKBACK_WINDOWS_HOURS`), and deployment modes (`STAGING` vs `PRODUCTION`) without code modifications.
2. **Refactored Fallback Preprocessing (`src/preprocessing.py`)**:
   - Fallback hierarchy for physical site extraction: `Alarm Source` -> `MO Name` -> `BBU Name` -> `NE Name` -> `Location Information`.
   - Extracts 4-7 char alphanumeric prefix (e.g. `ZBIY43`, `GLIND1`) as `site_id` while preserving `node_identifier`.
   - Granular alarm categorization into 7 flags: `OUTAGE_CELL_UNAVAILABLE`, `OUTAGE_CELL_OUTAGE`, `OUTAGE_CELL_FAULT`, `OUTAGE_SERVICE_UNAVAILABLE`, `BBU_ALARM`, `RRU_ALARM`, `POWER_ALARM`.
3. **Airflow Pipeline & Auto-Purge (`dags/mobitel_outage_dag.py`)**:
   - Task 1: `ingest_and_clean_data`
   - Task 2: `auto_purge_60_days` (runs `DELETE FROM master_alarms WHERE datetime(event_time) < datetime('now', '-60 days')`)
   - Task 3: `run_model_inference` (outputs predictions to `data/latest_predictions.csv`).
4. **Secure Admin Panel & Dual-Environment UI (`app.py`)**:
   - `Tab 4: 🔐 Admin Settings` protected by password authentication.
   - Live environment mode header badge: `[MODE: STAGING (TESTING)]` vs `[MODE: PRODUCTION (FIXED)]`.
   - Two distinct adjacent table columns: `Site ID (Short)` and `eNodeB / gNodeB Identifier`.

---

## 📊 Dataset & Feature Engineering Summary

- **Source Dataset**: 36 Huawei iManager U2000/NCE alarm log Excel files (`AlarmLogs20260803113740030_*.xlsx`).
- **Date Range**: **July 1, 2026 00:00:00 – July 31, 2026 23:59:59** (31 Full Days).
- **Total Raw Records Scanned**: **3,545,304** alarm log events across 185,440 network site locations.
- **Lookback Windows**: Past 6 hours $[T - 6\text{h}, T)$, 24 hours $[T - 24\text{h}, T)$, 14 days $[T - 336\text{h}, T)$, and 30 days $[T - 720\text{h}, T)$.
- **Prediction Window**: Next 2 hours $[T, T + 2\text{h})$.

---

## 🎯 Model Validation Results

- **Holdout Validation Period**: July 23, 2026 – July 31, 2026
- **PR-AUC Score**: **0.7925**
- **ROC-AUC Score**: **0.9611**
- **Recall @ Optimal Threshold**: **91.09%**

---

## 💻 Quickstart & Execution Instructions

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run Dynamic Feature Engineering & Model Training
python3 src/feature_engineering.py
python3 src/model.py

# 3. Launch Interactive Streamlit Dashboard (Port 8502)
streamlit run app.py --server.port 8502
```