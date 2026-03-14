import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# ─────────────────────────────────────────
# configuration
# ─────────────────────────────────────────

FLOW_CSV = "household_flow_365d.csv"
OUT_DIR = Path("model_artifacts")

WINDOW_MINUTES = 20
STRIDE_MINUTES = 5

# NEW: appliance threshold
APPLIANCE_FLOW_THRESH = 0.8

INTER_FRAC_THRESH = 0.02

N_ESTIMATORS = 300
RANDOM_STATE = 42

CUSUM_K = 0.01
CUSUM_H = 1.0

IF_THRESHOLD_PERCENTILE = 5


# ─────────────────────────────────────────
# feature extraction
# ─────────────────────────────────────────

def extract_window_features(window):

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

    rows = []

    for start in range(0, len(flow) - WINDOW_MINUTES + 1, STRIDE_MINUTES):

        window = flow[start:start + WINDOW_MINUTES]

        rows.append(extract_window_features(window))

    return np.vstack(rows)


# ─────────────────────────────────────────
# threshold calibration
# ─────────────────────────────────────────

def calibrate_if_threshold(scores):

    threshold = float(np.percentile(scores, IF_THRESHOLD_PERCENTILE))

    scale = float(abs(scores.min() - threshold))

    scale = max(scale, 1e-6)

    return threshold, scale


# ─────────────────────────────────────────
# training
# ─────────────────────────────────────────

def main():

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