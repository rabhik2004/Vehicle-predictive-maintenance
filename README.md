# AI/ML-Based Predictive Maintenance and Vehicle Health Monitoring System
Module 3 (Connectivity) Project - Option 1

## Goal
Predict a vehicle/component failure *before* it happens, convert the prediction into a
Vehicle Health Score, raise a maintenance alert, and explain the alert with XAI (SHAP).

## Pipeline (matches the assignment architecture)
Vehicle sensors -> preprocessing/feature engineering -> ML/DL model -> failure probability
-> vehicle health score -> maintenance alert (+ SHAP contributing factors)

| Stage | File |
|---|---|
| Sensor simulation (9 inputs from the brief) | `generate_data.py` |
| Preprocessing: rolling mean / std / slope (window 10) per vehicle | `features.py` |
| Models: Logistic Regression, Decision Tree, Random Forest, XGBoost | `train.py` |
| Deep models: LSTM / GRU | `lstm_model.py` |
| Health score, status, SHAP factors, report | `health.py` |

## Run
    pip install -r requirements.txt
    python train.py            # trains, evaluates, saves plots + model.joblib
    python lstm_model.py --arch lstm   # optional (needs PyTorch)

## Data
No real fleet dataset was supplied, so data is **simulated**: 400 vehicles, healthy ones stay near
baseline, ~55% degrade before failing (vibration, temperature and coolant rise; oil pressure and
battery voltage fall). Severity, lead time and per-sensor sensitivity vary per vehicle, with heavy
sensor noise. Label = failure within the next 20 readings. To use real data (e.g. NASA C-MAPSS or a
fleet CSV), replace `generate()` with a loader returning the same columns.

## Evaluation (held-out vehicles; split by vehicle to prevent leakage)
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.966 | 0.659 | 0.922 | 0.769 | 0.988 |
| Decision Tree | 0.961 | 0.627 | 0.890 | 0.736 | 0.905 |
| Random Forest | 0.970 | 0.712 | 0.865 | 0.781 | 0.986 |
| XGBoost | 0.970 | 0.710 | 0.850 | 0.774 | 0.986 |

Class imbalance handled with class weights / `scale_pos_weight`. Random Forest was best by F1 and
is used for the health monitor. Threshold can be lowered to favour recall, which matters more than
precision in maintenance (a missed failure costs more than an early inspection).

## Health score and alert logic
- Health = 100 x (0.5 x (1 - failure probability) + 0.5 x (1 - weighted sensor deviation))
- Status: HEALTHY (< 35% failure probability), WARNING (35-75%), CRITICAL (> 75%)
- Contributing factors: SHAP values summed per physical sensor; top positive contributors shown.

## Sample output (from a run)
    Vehicle Health: 69%              Vehicle Health: 30%
    Failure Probability: 55%         Failure Probability: 95%
    Status: WARNING                  Status: CRITICAL
    1. Elevated fuel consumption     1. Abnormal speed pattern
    2. Low battery voltage           2. Rising coolant temperature
    3. Abnormal RPM                  3. Abnormal RPM

## Limitations
- Results are on simulated data; real-world accuracy will differ.
- Vehicle speed appears as a "factor" because it acts as a load proxy that lets the model
  discount sensor values that are high only because of heavy driving.
- The health-score weights are hand-set design choices, not learned.
- `lstm_model.py` was not executed in the build environment (PyTorch could not be installed there).
