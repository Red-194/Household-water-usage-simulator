"""
Water Flow Simulation for Household Appliances

DESCRIPTION:
    Generates realistic minute-by-minute water usage data for household appliances
    based on Indian consumption standards. Produces synthetic 365-day datasets with
    authentic temporal patterns, appliance-specific behaviors, and sensor noise.

KEY FEATURES:
    - Appliance-aware event generation (showers, toilets, washing machines, etc.)
    - Realistic flow shape curves (trapezoid ramps, pulsed patterns)
    - Daily volume constraints (100-160 L/day per BIS IS 1172 standards)
    - Multi-attempt regeneration for volume validation
    - Additive sensor noise on non-zero flow bins
    - Sliding-window feature extraction for ML models

OUTPUT:
    - household_flow_365d.csv: Minute-level flow rates (1440 * 365 rows)
    - household_events_365d.csv: Event log with timing, duration, flow per appliance

DEPENDENCIES:
    - numpy, pandas: Data processing and math
    - json: Appliance configuration loading
"""

import json
import numpy as np
import pandas as pd
import os
import warnings

PRIORS_PATH = "all_appliances.json"
OUTPUT_DIR = "simulator_data"

DAYS = 365
MINUTES_PER_DAY = 1440
SECONDS_PER_MIN = 60
START_TS = 0

MAX_FLOW_LPM = 15.0

DAILY_MIN_L = 100
DAILY_MAX_L = 160

MAX_REGEN_ATTEMPTS = 10

NOISE_SIGMA = 0.03

WM_OFFSET = None

def sample_lognormal(shape, scale):
    """
    Sample a random value from a log-normal distribution.
    
    Used for realistic flow rates and durations across diverse appliances.
    Log-normal better captures the skewed nature of water usage (long tail of
    extended sessions).
    """
    return np.random.lognormal(mean=np.log(scale), sigma=shape)


def sample_start_minute(hour_probs):
    """
    Sample a random start minute within the day based on hourly usage patterns.
    
    Selects an hour using the appliance's hourly probability distribution,
    then adds random offset (0-59 minutes). Preserves realistic timing patterns
    while avoiding artificial gridding at hour boundaries.
    """
    return np.random.choice(24, p=hour_probs) * 60 + np.random.randint(60)


def bounded_duration(scale, min_s, max_s):
    """
    Sample duration from normal distribution with hard constraints.
    
    Generates durations with inherent variability (±25% standard deviation)
    while respecting appliance-specific min/max bounds. Used for quick-use
    appliances (washbasin, faucets) with relatively predictable usage times.
    """
    return np.clip(
        np.random.normal(loc=scale, scale=0.25 * scale),
        min_s,
        max_s
    )
def make_shape_curve(shape, shape_cfg, dur):
    """
    Generate a normalized flow shape curve for realistic appliance usage patterns.
    
    Simulates different water usage characteristics:
    - "trapezoid": Gradual ramp-up/ramp-down (realistic taps, showers)
    - "pulsed": Intermittent on-off pattern (washing machines, certain cycles)
    - default "step": Constant flow (simple model)
    
    Curve is normalized to mean=1.0 to preserve the sampled flow rate across
    different shape types, ensuring fair comparison and consistent volume control.
    """
    if shape == "trapezoid" and dur >= 4:
        ramp_s = shape_cfg.get("ramp_up_s", 5)
        fall_s = shape_cfg.get("ramp_down_s", 5)

        ramp_bins  = max(1, int(np.ceil(ramp_s / SECONDS_PER_MIN)))
        fall_bins  = max(1, int(np.ceil(fall_s / SECONDS_PER_MIN)))
        plateau_bins = max(0, dur - ramp_bins - fall_bins)

        if plateau_bins <= 0:
            return np.ones(dur)

        curve = np.concatenate([
            np.linspace(0.5, 1.0, ramp_bins),
            np.ones(plateau_bins),
            np.linspace(1.0, 0.5, fall_bins),
        ])

    elif shape == "pulsed" and dur >= 2:
        pulse = (np.arange(dur) % 4 < 2).astype(float)
        if pulse.sum() == 0:
            return np.ones(dur)
        curve = pulse

    else:
        return np.ones(dur)

    mean = curve.mean()
    return curve / mean if mean > 0 else np.ones(dur)

# =============================
# EVENT GENERATION
# =============================
def generate_events_for_day(appliance, day):
    """
    Generate water usage events for a specific appliance on a given day.
    
    Orchestrates realistic event generation by:
    1. Sampling daily event count from Poisson distribution (λ = expected events)
    2. Applying appliance-specific constraints (shower ≥1, toilet ≥2, WM once/week)
    3. Sampling start times using hourly probability distributions
    4. Sampling durations (fixed or random) within min/max constraints
    5. Sampling flow rates from log-normal distribution
    
    Produces event dictionaries that are later rendered into flow time series.
    """
    name      = appliance["appliance"]
    lam       = appliance["activation"]["events_per_day"]["lambda"]
    hour_probs = appliance["timing"]["start_hour"]["p"]

    # FIX 4: shower respects lambda with floor of 1
    if name == "shower":
        n_events = max(1, np.random.poisson(lam))

    # FIX 5: bidet floor removed
    elif name == "toilet":
        n_events = max(2, np.random.poisson(lam))
    elif name == "bidet":
        n_events = np.random.poisson(lam)

    elif name == "washingmachine":
        n_events = 1 if (day + WM_OFFSET) % 7 == 0 else 0
    else:
        n_events = np.random.poisson(lam)

    events = []

    for _ in range(n_events):
        start_min = day * MINUTES_PER_DAY + sample_start_minute(hour_probs)

        dur_cfg = appliance["duration"]

        if dur_cfg["type"] == "fixed":
            duration_s = dur_cfg["value"]
        elif name in ["washbasin", "kitchenfaucet", "bidet", "toilet"]:
            duration_s = bounded_duration(
                dur_cfg["scale"],
                appliance["constraints"]["min_duration_s"],
                appliance["constraints"]["max_duration_s"]
            )
        else:
            duration_s = sample_lognormal(dur_cfg["shape"], dur_cfg["scale"])

        duration_s = np.clip(
            duration_s,
            appliance["constraints"]["min_duration_s"],
            appliance["constraints"]["max_duration_s"]
        )

        # FIX 3: all appliances use JSON flow prior
        flow_cfg = appliance["flow"]["mean_flow"]
        mean_flow_ml_s = sample_lognormal(flow_cfg["shape"], flow_cfg["scale"])
        
        events.append({
            "appliance":       name,
            "start_min":       int(start_min),
            "duration_s":      float(duration_s),
            "mean_flow_ml_s":  float(mean_flow_ml_s),
            "shape":           appliance.get("shape", {}).get("type", "step"),
            "shape_cfg":       appliance.get("shape", {}),
        })

    return events

# =============================
# RENDER DAY
# =============================
def render_day(events, day):
    """
    Render a day's worth of minute-level flow data from usage events.
    
    Converts discrete events into continuous flow time series by:
    1. Initializing a zero flow array (1440 minutes)
    2. For each event: converting event flow/duration to minute grid
    3. Applying shape curves (trapezoid/pulsed) for realistic profiles
    4. Clipping flow to MAX_FLOW_LPM to respect physical constraints
    5. Accumulating overlapping appliance flows
    
    Output is passed to noise injection in the simulate() function.
    """
    flow = np.zeros(MINUTES_PER_DAY)

    for ev in events:
        start = ev["start_min"] - day * MINUTES_PER_DAY
        dur   = int(np.ceil(ev["duration_s"] / SECONDS_PER_MIN))
        lpm   = ev["mean_flow_ml_s"] * 60 / 1000

        end = min(start + dur, MINUTES_PER_DAY)
        actual_dur = end - start

        if actual_dur <= 0:
            continue

        # FIX 2: shape-aware rendering
        curve = make_shape_curve(ev["shape"], ev["shape_cfg"], actual_dur)

        flow[start:end] += lpm * curve
        flow[start:end]  = np.minimum(flow[start:end], MAX_FLOW_LPM)

    return flow

# =============================
# MAIN SIMULATION
# =============================
def simulate(priors, days, seed=42):
    """
    Run complete water flow simulation for specified number of days.
    
    ALGORITHM:
    1. Set random seed for reproducibility
    2. Initialize washing machine weekly offset (avoid RNG state corruption)
    3. For each day:
        a. Generate events for all appliances
        b. Render events to minute-level flows
        c. Check daily volume [DAILY_MIN_L, DAILY_MAX_L]
        d. Retry up to MAX_REGEN_ATTEMPTS if volume invalid
        e. Apply additive sensor noise (Gaussian, σ=NOISE_SIGMA*flow)
    4. Aggregate all flows and events into DataFrames
    5. Return flows with timestamps, and cleaned events (without shape_cfg)
    
    RETURNS:
        tuple: (flow_df, events_df)
            - flow_df: columns [timestamp (seconds), flow_lpm]
            - events_df: columns [appliance, start_min, duration_s, mean_flow_ml_s, shape]
    """
    np.random.seed(seed)

    global WM_OFFSET
    WM_OFFSET = np.random.randint(7)

    flow       = np.zeros(days * MINUTES_PER_DAY)
    all_events = []

    for day in range(days):
        # FIX 1: remove global scaling — let volume emerge, regenerate extremes
        day_flow = np.zeros(MINUTES_PER_DAY)  # Initialize with zeros
        day_events = []
        
        for attempt in range(MAX_REGEN_ATTEMPTS):
            day_events = []
            for appliance in priors:
                day_events.extend(generate_events_for_day(appliance, day))

            day_flow   = render_day(day_events, day)
            day_volume = day_flow.sum()

            if DAILY_MIN_L <= day_volume <= DAILY_MAX_L:
                break

            if attempt == MAX_REGEN_ATTEMPTS - 1:
                warnings.warn(
                    f"Day {day}: volume {day_volume:.1f} L outside "
                    f"[{DAILY_MIN_L}, {DAILY_MAX_L}] after "
                    f"{MAX_REGEN_ATTEMPTS} attempts. Accepting anyway."
                )

        # FIX 7: additive sensor noise on non-zero bins
        noisy = day_flow.copy()
        nonzero = noisy > 0
        noisy[nonzero] += np.random.normal(
            0.0, NOISE_SIGMA * noisy[nonzero], nonzero.sum()
        )
        noisy = np.maximum(noisy, 0.0)

        flow[day * MINUTES_PER_DAY:(day + 1) * MINUTES_PER_DAY] = noisy
        all_events.extend(day_events)

    timestamps = START_TS + np.arange(len(flow)) * SECONDS_PER_MIN

    # drop shape_cfg from events df (not serialisation-friendly)
    clean_events = [
        {k: v for k, v in e.items() if k != "shape_cfg"}
        for e in all_events
    ]

    return (
        pd.DataFrame({"timestamp": timestamps, "flow_lpm": flow}),
        pd.DataFrame(clean_events)
    )

# =============================
# DAILY AGGREGATION
# =============================
def compute_daily_usage(df):
    """
    Calculate daily water usage from minute-by-minute flow data.
    
    Args:
        df (pd.DataFrame): DataFrame with 'timestamp' and 'flow_lpm' columns
    
    Returns:
        pd.Series: Daily usage in liters per day, indexed by day number
    """
    df = df.copy()
    df["day"] = df["timestamp"] // 86400
    return df.groupby("day")["flow_lpm"].sum()


if __name__ == "__main__":
    """
    Standalone execution: Generates 365-day synthetic water usage dataset.
    
    OUTPUT:
        - simulator_data/household_flow_365d.csv (525,600 rows)
        - simulator_data/household_events_365d.csv (event log)
    PRINTS:
        - Event counts by appliance
        - Daily usage statistics (mean, min, max)
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(PRIORS_PATH) as f:
        priors = json.load(f)["appliances"]

    df_flow, df_events = simulate(priors, DAYS)

    df_flow.to_csv(f"{OUTPUT_DIR}/household_flow_365d.csv", index=False)
    df_events.to_csv(f"{OUTPUT_DIR}/household_events_365d.csv", index=False)

    daily = compute_daily_usage(df_flow)

    print("✅ Simulation complete")
    print(f"Days simulated: {DAYS}")

    print("\nEvents per appliance:")
    print(df_events["appliance"].value_counts())

    print("\nDaily water usage (L/day):")
    print(daily.round(1).to_string())

    print("\nSummary:")
    print(f"Mean: {daily.mean():.1f} L/day")
    print(f"Min : {daily.min():.1f} L/day")
    print(f"Max : {daily.max():.1f} L/day")