# Telecom CellDown Prediction (Mobitel Alarm Fault Prediction)

This repository contains the machine learning pipeline and analysis for predicting cell down faults and telecom network alarm events.

## Project Structure

```
Telecom-CellDown-Prediction/
│
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
│
├── notebooks/
│   ├── 01_Data_Loading.ipynb
│   ├── 02_Data_Preprocessing.ipynb
│   ├── 03_EDA.ipynb
│   ├── 04_Feature_Engineering.ipynb
│   ├── 05_Model_Training.ipynb
│   ├── 06_Model_Evaluation.ipynb
│   └── 07_Prediction.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── models/
├── figures/
└── reports/
```

## Setup & Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run notebooks in `notebooks/` or execute pipeline scripts in `src/`.