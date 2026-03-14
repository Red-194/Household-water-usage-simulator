import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import lognorm


# -------------------------
# Paths
# -------------------------
EVENTS_DIR = Path("events_merged")   # <-- use merged events now
PRIORS_DIR = Path("priors")
PRIORS_DIR.mkdir(exist_ok=True)


# -------------------------
# Utilities
# -------------------------

def positive_only(data, eps=1e-6):
    data = np.asarray(data)
    return data[data > eps]


def report_filtering(name, raw, filtered):
    dropped = len(raw) - len(filtered)
    if dropped > 0:
        print(f"[WARN] {name}: dropped {dropped}/{len(raw)} non-positive samples")


def fit_lognormal(data, min_samples=10):
    data = positive_only(data)

    if len(data) < min_samples:
        return {
            "type": "fixed",
            "value": float(np.mean(data)) if len(data) > 0 else 0.0
        }

    shape, loc, scale = lognorm.fit(data, floc=0)
    return {
        "type": "lognormal",
        "shape": float(shape),
        "scale": float(scale)
    }


def fit_poisson(events_per_day):
    return {
        "type": "poisson",
        "lambda": float(np.mean(events_per_day))
    }


def fit_categorical(prob_vector):
    return {
        "type": "categorical",
        "p": prob_vector.tolist()
    }


# -------------------------
# Dataset span (GLOBAL)
# -------------------------

def compute_dataset_span(csv_files):
    min_ts = None
    max_ts = None

    for csv_path in csv_files:
        df = pd.read_csv(csv_path, usecols=["start_ts", "end_ts"])
        min_ts = df["start_ts"].min() if min_ts is None else min(min_ts, df["start_ts"].min())
        max_ts = df["end_ts"].max() if max_ts is None else max(max_ts, df["end_ts"].max())

    span_seconds = max_ts - min_ts
    span_days = span_seconds / 86400.0

    return {
        "start_ts": int(min_ts),
        "end_ts": int(max_ts),
        "span_seconds": float(span_seconds),
        "span_days": float(span_days),
        "span_days_ceil": int(np.ceil(span_days))
    }


# -------------------------
# Prior extraction
# -------------------------

def extract_priors(csv_path: Path, dataset_span):
    appliance = csv_path.stem.replace("_events", "")
    df = pd.read_csv(csv_path)

    # ---- timestamps ----
    df["start_dt"] = pd.to_datetime(df["start_ts"], unit="s", utc=True)
    df["date"] = df["start_dt"].dt.date
    df["hour"] = df["start_dt"].dt.hour

    priors = {
        "appliance": appliance,
        "version": "1.0",
        "dataset_span": dataset_span
    }

    # -------------------------
    # Activation: events per day
    # -------------------------
    # include zero-event days
    all_days = pd.date_range(
        start=pd.to_datetime(dataset_span["start_ts"], unit="s", utc=True).date(),
        end=pd.to_datetime(dataset_span["end_ts"], unit="s", utc=True).date(),
        freq="D"
    )

    daily_counts = (
        df.groupby("date").size()
        .reindex(all_days.date, fill_value=0)
        .values
    )

    priors["activation"] = {
        "events_per_day": fit_poisson(daily_counts)
    }

    # -------------------------
    # Timing: hour-of-day
    # -------------------------
    hour_counts = df["hour"].value_counts().sort_index()
    hour_probs = hour_counts.reindex(range(24), fill_value=0).values
    hour_probs = hour_probs / hour_probs.sum()

    priors["timing"] = {
        "start_hour": fit_categorical(hour_probs)
    }

    # -------------------------
    # Duration
    # -------------------------
    durations_raw = df["duration_s"].values
    durations = positive_only(durations_raw)
    report_filtering("duration", durations_raw, durations)

    if len(durations) == 0 or np.std(durations) < 1e-3:
        priors["duration"] = {
            "type": "fixed",
            "value": float(np.mean(durations)) if len(durations) else 0.0,
            "unit": "seconds"
        }
    else:
        priors["duration"] = {
            **fit_lognormal(durations),
            "unit": "seconds"
        }

    # -------------------------
    # Flow
    # -------------------------
    mean_flow_raw = df["mean_flow_ml_s"].values
    mean_flow = positive_only(mean_flow_raw)
    report_filtering("mean_flow", mean_flow_raw, mean_flow)

    peak_flow_raw = df["peak_flow_ml_s"].values
    peak_flow = positive_only(peak_flow_raw)
    report_filtering("peak_flow", peak_flow_raw, peak_flow)

    priors["flow"] = {
        "mean_flow": {
            **fit_lognormal(mean_flow),
            "unit": "ml/s"
        },
        "peak_flow": {
            **fit_lognormal(peak_flow),
            "unit": "ml/s"
        }
    }

    # -------------------------
    # Shape heuristics
    # -------------------------
    if appliance in {"toilet", "kitchenfaucet", "washbasin"}:
        shape = {"type": "step"}
    elif appliance in {"shower", "bidet"}:
        shape = {"type": "trapezoid", "ramp_up_s": 5, "ramp_down_s": 5}
    else:
        shape = {"type": "pulsed"}

    priors["shape"] = shape

    # -------------------------
    # Constraints
    # -------------------------
    priors["constraints"] = {
        "volume_relation": "volume = mean_flow * duration",
        "min_duration_s": float(np.min(durations)) if len(durations) else 0.0,
        "max_duration_s": float(np.max(durations)) if len(durations) else 0.0
    }

    return priors


# -------------------------
# Main
# -------------------------

def main():
    csv_files = list(EVENTS_DIR.glob("*.csv"))
    dataset_span = compute_dataset_span(csv_files)

    print(
        f"[INFO] Dataset span: "
        f"{dataset_span['span_days']:.2f} days "
        f"({dataset_span['start_ts']} → {dataset_span['end_ts']})"
    )

    for csv_path in csv_files:
        priors = extract_priors(csv_path, dataset_span)

        out_path = PRIORS_DIR / f"{priors['appliance']}.json"
        with open(out_path, "w") as f:
            json.dump(priors, f, indent=2)

        print(f"[OK] Priors extracted for {priors['appliance']}")


if __name__ == "__main__":
    main()
