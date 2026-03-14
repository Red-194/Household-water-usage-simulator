import pandas as pd
import numpy as np

# ======================================================
# CONFIG
# ======================================================
CSV_PATH = "simulator_data/household_flow_365d_updated.csv"

# ======================================================
# LOAD DATA
# ======================================================
df = pd.read_csv(CSV_PATH)

# ------------------------------------------------------
# TIMESTAMP PARSING (UNIX SECONDS, 60s resolution)
# ------------------------------------------------------
df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
df = df.sort_values("timestamp")

flow = df["flow_lpm"].astype(float)

print(f"\nLoaded {len(df):,} samples")
print(f"Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")

# ======================================================
# DAILY USAGE
# ======================================================
print("\n=== DAILY WATER USAGE (L/day) ===")
daily = df.set_index("timestamp")["flow_lpm"].resample("D").sum()
print(daily.tail(10))

mean = daily.mean()
std = daily.std()
cv = std / mean

print("\n=== SUMMARY ===")
print(f"Mean      : {mean:.2f}")
print(f"Std       : {std:.2f}")
print(f"Min       : {daily.min():.2f}")
print(f"10th pct  : {np.percentile(daily, 10):.2f}")
print(f"Median    : {np.percentile(daily, 50):.2f}")
print(f"90th pct  : {np.percentile(daily, 90):.2f}")
print(f"Max       : {daily.max():.2f}")

# ======================================================
# VARIABILITY CHECK (HOUSEHOLD)
# ======================================================
print("\n=== VARIABILITY CHECK ===")
print(f"Coeff. of variation: {cv:.3f}")
print("Expected (household):")
print("  Typical : 0.10 – 0.30")
print("  Accept  : 0.08 – 0.40")

# ======================================================
# NIGHT BASELINE
# ======================================================
print("\n=== NIGHT BASELINE (0–5 AM) ===")
night = df[df["timestamp"].dt.hour < 5]["flow_lpm"]

print(f"Median   : {night.median():.3f} L/min")
print(f"95th pct : {np.percentile(night, 95):.3f} L/min")
print("Expected:")
print("  Median < 0.1 L/min")
print("  95th pct < 2.0 L/min")

# ======================================================
# DAY / NIGHT RATIO
# ======================================================
print("\n=== PEAK / NIGHT RATIO ===")
morning = df[
    (df["timestamp"].dt.hour >= 6) &
    (df["timestamp"].dt.hour <= 9)
]["flow_lpm"]

night_mean = night.mean()
ratio = morning.mean() / (night_mean + 1e-6)

print(f"Morning avg: {morning.mean():.2f}")
print(f"Night avg  : {night_mean:.2f}")
print(f"Ratio      : {ratio:.2f}x")
print("Expected: 4 – 12x")

# ======================================================
# FLOW DISTRIBUTION
# ======================================================
print("\n=== FLOW DISTRIBUTION ===")
desc = flow.describe(percentiles=[0.5, 0.9, 0.99])
print(desc)

unique_ratio = flow.nunique() / len(flow)
print(f"\nUnique value ratio: {unique_ratio:.4f}")
print("Expected: > 0.01")

# ======================================================
# AUTOCORRELATION
# ======================================================
print("\n=== AUTOCORRELATION ===")
lag1 = flow.autocorr(lag=1)
lag5 = flow.autocorr(lag=5)

print(f"lag-1: {lag1:.3f}")
print(f"lag-5: {lag5:.3f}")
print("Expected: lag-1 < 0.6")

# ======================================================
# ZERO / NEAR-ZERO FLOW
# ======================================================
print("\n=== ZERO / NEAR-ZERO FLOW ===")
zero_frac = np.mean(flow < 0.05)
print(f"Fraction near zero (<0.05 LPM): {zero_frac:.3f}")
print("Expected: > 0.20")

# ======================================================
# FINAL VERDICT
# ======================================================
fail = 0
warn = 0

# Hard failures (physics violations)
if night.median() > 0.1:
    fail += 1
if ratio < 3.5:
    fail += 1
if unique_ratio < 0.01:
    fail += 1
if lag1 > 0.6:
    fail += 1

# Warnings (plausible but notable)
if cv < 0.08:
    warn += 1
if cv > 0.40:
    warn += 1
if ratio > 12.0:
    warn += 1

print("\n=== VERDICT ===")
if fail == 0 and warn == 0:
    print("✅ DATA PASSES HOUSEHOLD SANITY CHECK")
elif fail == 0:
    print(f"⚠️  DATA PASSES WITH WARNINGS ({warn})")
else:
    print(f"❌ DATA FAILS ({fail} critical issues, {warn} warnings)")
