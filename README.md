# 4G Telecom Tower Cell Unavailability Outage Prediction

An end-to-end autonomous data engineering and machine learning framework that processes raw Huawei 4G telecom alarm logs, engineers site-level 6-hour sliding window features with a 2-hour forward lookahead, and trains tabular GBDT classifiers to predict cell unavailability outages.

---

## 1. Dataset Summary & Date Ranges

- **Source Dataset**: 36 Huawei iManager U2000/NCE alarm log Excel files (`AlarmLogs20260803113740030_*.xlsx`).
- **Date Range**: **July 1, 2026 00:00:00 – July 31, 2026 23:59:59** (31 Full Days).
- **Total Raw Records Scanned**: **3,545,304** alarm log events across 3,229+ network locations.
- **Tower Subset Extraction (`july_2026_subset_dataset.csv`)**:
  - **Top 20 Outage Towers**: Selected based on highest frequency of target outage alarms (`CELL UNAVAILABLE` / `Cell Unavailable`).
  - **15 Healthy Towers**: Selected from towers experiencing exactly zero outage alarms throughout July 2026.
  - **Subset Total**: **916,223** raw alarm log rows extracted across 35 representative towers.

---

## 2. Feature Engineering Methodology

A sliding window algorithm evaluates each tower site chronologically across July 2026 in **2-hour step increments**:

$$T \in \{ \text{2026-07-01 00:00}, \text{2026-07-01 02:00}, \dots, \text{2026-07-31 22:00} \}$$

### Window Definitions
- **Lookback Window ($X$)**: Past 6 hours relative to timestamp $T$, i.e., $[T - 6\text{h}, T)$.
- **Prediction Window ($Y$)**: Next 2 hours relative to timestamp $T$, i.e., $[T, T + 2\text{h})$.

### Feature Matrix Schema (`telecom_features_july_2026.csv`)

| Feature Name | Type | Description |
| :--- | :--- | :--- |
| `site_id` | String | Unique 4G Tower / eNodeB identifier |
| `window_timestamp` | Datetime | Start timestamp $T$ of prediction window |
| `total_alarms_6h` | Integer | Total count of all alarms occurring in $[T-6\text{h}, T)$ |
| `critical_alarms_6h` | Integer | Count of Critical severity alarms occurring in $[T-6\text{h}, T)$ |
| `major_alarms_6h` | Integer | Count of Major severity alarms occurring in $[T-6\text{h}, T)$ |
| `rru_alarms_6h` | Integer | Count of RF, RRU, or VSWR failure keyword alarms in $[T-6\text{h}, T)$ |
| `bbu_alarms_6h` | Integer | Count of BBU, Board, Subrack, or Transmission keyword alarms in $[T-6\text{h}, T)$ |
| `target_outage_next_2h` | Binary ($0/1$) | Target Label: $1$ if `CELL UNAVAILABLE` alarm occurs in $[T, T+2\text{h})$, else $0$ |

**Total Feature Rows**: **13,020** sliding window samples ($372$ windows $\times 35$ sites).  
**Positive Outage Target Rate**: **11.75%** ($1,530$ positive outage windows).

---

## 3. Model Training & Evaluation Results

### Strict Chronological Split (No Data Leakage)
- **Training Period**: July 1, 2026 – July 22, 2026 ($9,240$ samples, $1,058$ positive outage instances)
- **Holdout Test Period**: July 23, 2026 – July 31, 2026 ($3,780$ samples, $472$ positive outage instances)

### Test Holdout Metrics (July 23–31, 2026)

| Metric | Score |
| :--- | :--- |
| **PR-AUC (Precision-Recall Area Under Curve)** | **0.6570** |
| **ROC-AUC (Receiver Operating Characteristic)** | **0.8589** |
| **Precision (at 0.5 Decision Threshold)** | **0.5228** |
| **Recall (at 0.5 Decision Threshold)** | **0.5826** |
| **Overall Model Accuracy** | **88.04%** |

---

## 4. Top Predictive Alarm Features

Permutation Importance analysis on the holdout test set reveals the relative predictive power of each alarm feature:

| Rank | Feature | Importance Score (PR-AUC Impact) | Key Drivers |
| :---: | :--- | :---: | :--- |
| **1** | `critical_alarms_6h` | **0.3502** | Severe physical/hardware faults triggering immediate cascade |
| **2** | `rru_alarms_6h` | **0.2780** | RF unit, VSWR antenna impedance, and radio transceivers |
| **3** | `bbu_alarms_6h` | **0.0751** | Baseband unit board, CPRI interface errors, subrack issues |
| **4** | `major_alarms_6h` | **0.0723** | Major transmission path and IP path faults |
| **5** | `total_alarms_6h` | **0.0271** | General alarm storm activity volume |

---

## 5. Repository Directory Structure

```
.
├── README.md                          # Project documentation and evaluation report
├── july_2026_subset_dataset.csv       # Extracted 35-tower raw alarm subset (916,223 rows)
├── telecom_features_july_2026.csv     # Engineered 6h/2h sliding-window feature matrix (13,020 rows)
├── requirements.txt                   # Dependency environment specification
├── figures/                           # Generated evaluation plots
│   ├── pr_curve.png                   # Precision-Recall evaluation curve
│   └── feature_importance.png         # Feature importance bar chart
├── models/                            # Saved trained model artifacts
│   └── outage_prediction_rf.pkl       # Trained Random Forest GBDT classifier
└── src/                               # Python source modules
    ├── preprocessing.py               # Data scanning & subset creation module
    ├── feature_engineering.py         # 6h/2h sliding window feature builder
    ├── train.py                       # Chronological split model training module
    ├── evaluate.py                    # Test holdout evaluation & metrics calculation
    └── predict.py                     # Inference pipeline for new alarm windows
```

---

## 6. Execution Instructions

To execute the pipeline from scratch:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run Data Scanning & Subset Extraction (Step 1)
python3 src/preprocessing.py

# 3. Generate Sliding-Window Features (Step 2)
python3 src/feature_engineering.py

# 4. Train Outage Classifier (Step 3)
python3 src/train.py

# 5. Evaluate Model on Holdout Test Set (Step 3)
python3 src/evaluate.py

# 6. Run Inference Sample
python3 src/predict.py
```