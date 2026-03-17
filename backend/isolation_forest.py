"""
Anomaly Detection Model Training: Isolation Forest

DESCRIPTION:
    Trains an Isolation Forest-based anomaly detector for water leak detection.
    Processes 365-day household flow data, extracts sliding-window features,
    normalizes them, and calibrates decision thresholds for production use.

WORKFLOW:
    1. Load household_flow_365d.csv (525,600 minute records)
    2. Extract 20-minute sliding windows with 5-minute stride (~105k windows)
    3. Compute 5 statistical features per window (see extract_window_features)
    4. Standardize features with StandardScaler
    5. Train Isolation Forest (300 trees, auto contamination)
    6. Calibrate threshold at 5th percentile (95% normal data below)
    7. Save artifacts: model, scaler, calibration JSON

FEATURE SET (per 20-minute window):
    - mnf: 10th percentile of non-zero flows (minimum normal flow baseline)
    - inter_mean: Mean flow below appliance threshold (leak baseline)
    - inter_frac: Fraction of low-flow samples above noise floor
    - mean_flow: Average window flow
    - inter_std: Standard deviation of low-flow baseline

KEY PARAMETERS:
    - WINDOW_MINUTES: 20 (detection window size)
    - STRIDE_MINUTES: 5 (75% overlap between windows)
    - APPLIANCE_FLOW_THRESH: 0.8 LPM (discriminates appliances from anomalies)
    - N_ESTIMATORS: 300 (Isolation Forest ensemble size)
    - IF_THRESHOLD_PERCENTILE: 5 (sets ~95% normal acceptance rate)

OUTPUT ARTIFACTS:
    - if_model.pkl: Trained classifier
    - if_scaler.pkl: Feature normalizer
    - if_calibration.json: Threshold + parameters for detector

DEPENDENCIES:
    - scikit-learn: IsolationForest, StandardScaler
    - pandas, numpy: Data processing
"""

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


FLOW_CSV = "household_flow_365d.csv"
OUT_DIR = Path("model_artifacts")

WINDOW_MINUTES = 20
STRIDE_MINUTES = 5
APPLIANCE_FLOW_THRESH = 0.8
INTER_FRAC_THRESH = 0.02

N_ESTIMATORS = 300
RANDOM_STATE = 42

CUSUM_K = 0.01
CUSUM_H = 1.0

IF_THRESHOLD_PERCENTILE = 5


def extract_window_features(window):
    """
    Extract 5 statistical features from a 20-minute flow window.
    
    Computes features designed to detect sustained low-flow anomalies (leaks)
    versus normal appliance usage. Features capture baseline inter-appliance
    behavior and window-level aggregates.
    
    OUTPUT FEATURES:
        1. mnf: 10th percentile of non-zero flows (baseline minimum normal)
        2. inter_mean: Mean of flows below APPLIANCE_FLOW_THRESH (leak baseline)
        3. inter_frac: Fraction of low flows above INTER_FRAC_THRESH (activity)
        4. mean_flow: Average flow across entire window
        5. inter_std: Std dev of low-flow baseline (pattern variance)
    
    Returns: float32 array for sklearn compatibility
    """

    # inter-event = anything below appliance level
    inter = window[window < APPLIANCE_FLOW_THRESH]

    nonzero = window[window > 0]

    if len(nonzero) > 0:
        mnf = float(np.percentile(nonzero, 10))
    else:
        mnf = 0.0

    inter_mean = float(inter.mean()) if len(inter) > 0 else 0.0
    inter_frac = float((inter > INTER_FRAC_THRESH).mean()) if len(inter) > 0 else 0.0
    inter_std = float(inter.std()) if len(inter) > 0 else 0.0

    mean_flow = float(window.mean())

    return np.array(
        [mnf, inter_mean, inter_frac, mean_flow, inter_std],
        dtype=np.float32
    )


def build_feature_matrix(flow):
    """
    Build feature matrix from complete time series using sliding windows.
    
    Extracts features at every 5-minute stride across entire flow dataset.
    Creates ~105,000 windows from 365-day flow data with 75% overlap,
    providing dense feature coverage for robust model training.
    
    OUTPUT SHAPE:
        (n_windows, 5) where n_windows ≈ (525600 - 20) / 5 + 1 ≈ 105,116
    """
    rows = []

    for start in range(0, len(flow) - WINDOW_MINUTES + 1, STRIDE_MINUTES):

        window = flow[start:start + WINDOW_MINUTES]

        rows.append(extract_window_features(window))

    return np.vstack(rows)


def calibrate_if_threshold(scores):
    """
    Calibrate Isolation Forest decision threshold from training anomaly scores.
    
    Sets threshold at 5th percentile of training scores, ensuring ~95% of
    normal behavior is classified as normal (below threshold). Computes a
    scale factor to normalize distances for the detector's hybrid scoring.
    
    OUTPUT:
        - threshold: 5th percentile (lower scores = more anomalous)
        - scale: Normalized range from minimum to threshold (for [0,1] scoring)
    
    PHYSICS:
        - Isolation Forest assigns negative scores to normal, positive to anomalies
        - Scale ∈ [min_score, threshold] maps to anomaly distance ∈ [0, 1]
    """

    threshold = float(np.percentile(scores, IF_THRESHOLD_PERCENTILE))

    scale = float(abs(scores.min() - threshold))

    scale = max(scale, 1e-6)

    return threshold, scale


def main():
    """
    Execute complete Isolation Forest training pipeline.
    
    STEPS:
    1. Load CSV flow data (525,600 minutes)
    2. Extract 105k+ sliding-window features
    3. Standardize features with StandardScaler
    4. Train Isolation Forest with 300 trees
    5. Score training data
    6. Calibrate decision threshold at 5th percentile
    7. Persist all artifacts to disk
    
    OUTPUT FILES (in model_artifacts/):
        - if_model.pkl: Trained Isolation Forest
        - if_scaler.pkl: StandardScaler fitted on training features
        - if_calibration.json: Threshold, parameters for detector
    """

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(FLOW_CSV)

    flow = df["flow_lpm"].values.astype(np.float32)

    X = build_feature_matrix(flow)

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    clf = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination="auto",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    clf.fit(X_scaled)

    scores = clf.decision_function(X_scaled)

    if_threshold, if_score_scale = calibrate_if_threshold(scores)

    calibration = {
        "window_minutes": WINDOW_MINUTES,
        "stride_minutes": STRIDE_MINUTES,
        "appliance_flow_thresh": APPLIANCE_FLOW_THRESH,
        "cusum_k": CUSUM_K,
        "cusum_h": CUSUM_H,
        "if_threshold": round(if_threshold,6),
        "if_score_scale": round(if_score_scale,6)
    }

    with open(OUT_DIR / "if_model.pkl", "wb") as f:
        pickle.dump(clf, f)

    with open(OUT_DIR / "if_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    with open(OUT_DIR / "if_calibration.json", "w") as f:
        json.dump(calibration, f, indent=2)

    print("Artifacts written to:", OUT_DIR)
    print(json.dumps(calibration, indent=2))


if __name__ == "__main__":
    main()