"""
Water Flow Simulation for Household Appliances

This module simulates realistic water usage patterns for household appliances
based on Indian consumption standards. It generates minute-by-minute flow data
with realistic usage patterns, timing distributions, and sensor noise.

Author: Water Detection Simulation System
"""

import json
import numpy as np
import pandas as pd
import os
import warnings

# File paths and output configuration
PRIORS_PATH = "all_appliances.json"  # Path to appliance configuration JSON
OUTPUT_DIR = "simulator_data"  # Directory for simulation output files

# Simulation parameters
DAYS = 365  # Number of days to simulate
MINUTES_PER_DAY = 1440  # Minutes in a day (24 * 60)
SECONDS_PER_MIN = 60  # Seconds per minute
START_TS = 0  # Starting timestamp for simulation

# Physical constraints
MAX_FLOW_LPM = 15.0  # Maximum flow rate in liters per minute

# Daily water budget (India, single-person well-served household)
# Lower bound: BIS IS 1172 / MoHUA recommended minimum (135 LPCD)
# Upper bound: comfortable urban usage above BIS norm
DAILY_MIN_L = 100  # Minimum daily water usage in liters
DAILY_MAX_L = 160  # Maximum daily water usage in liters

MAX_REGEN_ATTEMPTS = 10  # Maximum attempts to generate valid daily usage

# Sensor noise (multiplicative std dev)
NOISE_SIGMA = 0.03  # Standard deviation for sensor noise as fraction of signal

# Weekly appliance offsets — generated inside simulate() to avoid
# corrupting the random state when this module is imported elsewhere
WM_OFFSET = None  # Washing machine weekly offset (day of week)

# =============================
# HELPER FUNCTIONS
# =============================

def sample_lognormal(shape, scale):
    """
    Sample from a log-normal distribution.
    
    Args:
        shape (float): Shape parameter (sigma in log-normal)
        scale (float): Scale parameter (median of the distribution)
    
    Returns:
        float: Random sample from log-normal distribution
    """
    return np.random.lognormal(mean=np.log(scale), sigma=shape)


def sample_start_minute(hour_probs):
    """
    Sample a random start minute within a day based on hourly probabilities.
    
    Args:
        hour_probs (list): Probability distribution over 24 hours
    
    Returns:
        int: Random minute of the day (0-1439)
    """
    return np.random.choice(24, p=hour_probs) * 60 + np.random.randint(60)


def bounded_duration(scale, min_s, max_s):
    """
    Sample duration from normal distribution with bounds.
    
    Args:
        scale (float): Mean duration in seconds
        min_s (float): Minimum allowed duration
        max_s (float): Maximum allowed duration
    
    Returns:
        float: Bounded duration in seconds
    """
    return np.clip(
        np.random.normal(loc=scale, scale=0.25 * scale),
        min_s,
        max_s
    )

# =============================
# SHAPE CURVES
# =============================
def make_shape_curve(shape, shape_cfg, dur):
    """
    Generate a normalized flow shape curve for appliance usage patterns.
    
    Creates realistic flow patterns like trapezoids (gradual start/stop) or
    pulsed patterns (intermittent flow). The curve is normalized to mean=1.0
    to preserve the sampled flow rate.
    
    Args:
        shape (str): Shape type ('trapezoid', 'pulsed', or default 'step')
        shape_cfg (dict): Shape configuration parameters
        dur (int): Duration in minutes
    
    Returns:
        np.ndarray: Normalized flow curve with mean=1.0
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
    
    This function creates realistic usage events based on appliance-specific
    patterns, including frequency, timing, duration, and flow characteristics.
    
    Args:
        appliance (dict): Appliance configuration with usage patterns
        day (int): Day number (0-based) for simulation
    
    Returns:
        list: List of event dictionaries containing usage details
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
    Render a day's worth of water flow data from usage events.
    
    Converts water usage events into minute-by-minute flow rates,
    applying shape curves and flow limits.
    
    Args:
        events (list): List of water usage event dictionaries
        day (int): Day number for calculating relative timing
    
    Returns:
        np.ndarray: Array of flow rates for each minute of the day (LPM)
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
    Run the complete water flow simulation for specified number of days.
    
    Generates realistic water usage patterns with proper daily volumes,
    sensor noise, and appliance scheduling patterns.
    
    Args:
        priors (list): List of appliance configuration dictionaries
        days (int): Number of days to simulate
        seed (int): Random seed for reproducible results
    
    Returns:
        tuple: (flow_dataframe, events_dataframe)
            - flow_dataframe: Minute-by-minute flow data with timestamps
            - events_dataframe: Individual usage events with details
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

# =============================
# RUN
# =============================
if __name__ == "__main__":
    """
    Main execution block for running water flow simulation.
    
    Generates a 365-day simulation, saves data to CSV files,
    and prints summary statistics.
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