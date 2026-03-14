import csv
import os
from pathlib import Path

# ----------------------------
# CONFIG
# ----------------------------
EVENTS_DIR = Path("events")
OUTPUT_DIR = Path("events_merged")
OUTPUT_DIR.mkdir(exist_ok=True)

MERGE_WINDOWS = {
    "shower": 120,
    "washingmachine": 180,
    "kitchenfaucet": 0,
    "washbasin": 0,
    "bidet": 0,
    "toilet": 0,
    "dishwasher30": 0,
}

# ----------------------------
# MERGE FUNCTION
# ----------------------------
def merge_events(events, gap_threshold):
    if not events or gap_threshold == 0:
        return events

    merged = []
    current = events[0].copy()

    for nxt in events[1:]:
        gap = nxt["start_ts"] - current["end_ts"]

        if gap <= gap_threshold:
            # merge
            current["end_ts"] = nxt["end_ts"]
            current["total_volume_ml"] += nxt["total_volume_ml"]
            current["peak_flow_ml_s"] = max(
                current["peak_flow_ml_s"], nxt["peak_flow_ml_s"]
            )
        else:
            # finalize current
            duration = current["end_ts"] - current["start_ts"] + 1
            current["duration_s"] = duration
            current["mean_flow_ml_s"] = (
                current["total_volume_ml"] / duration if duration > 0 else 0
            )
            merged.append(current)
            current = nxt.copy()

    # finalize last
    duration = current["end_ts"] - current["start_ts"] + 1
    current["duration_s"] = duration
    current["mean_flow_ml_s"] = (
        current["total_volume_ml"] / duration if duration > 0 else 0
    )
    merged.append(current)

    return merged

# ----------------------------
# MAIN LOOP
# ----------------------------
for csv_file in EVENTS_DIR.glob("*_events.csv"):
    appliance = csv_file.stem.replace("_events", "")
    gap = MERGE_WINDOWS.get(appliance)

    if gap is None:
        print(f"[SKIP] No merge rule for {appliance}")
        continue

    # load events
    events = []
    with open(csv_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            events.append({
                "event_id": int(row["event_id"]),
                "start_ts": int(row["start_ts"]),
                "end_ts": int(row["end_ts"]),
                "duration_s": int(row["duration_s"]),
                "mean_flow_ml_s": float(row["mean_flow_ml_s"]),
                "peak_flow_ml_s": float(row["peak_flow_ml_s"]),
                "total_volume_ml": float(row["total_volume_ml"]),
            })

    before = len(events)
    merged = merge_events(events, gap)
    after = len(merged)

    # write merged events
    out_file = OUTPUT_DIR / csv_file.name
    with open(out_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "event_id",
            "start_ts",
            "end_ts",
            "duration_s",
            "mean_flow_ml_s",
            "peak_flow_ml_s",
            "total_volume_ml"
        ])

        for i, ev in enumerate(merged):
            writer.writerow([
                i,
                ev["start_ts"],
                ev["end_ts"],
                ev["duration_s"],
                round(ev["mean_flow_ml_s"], 3),
                round(ev["peak_flow_ml_s"], 3),
                round(ev["total_volume_ml"], 3),
            ])

    print(f"[{appliance}] {before} → {after} (gap={gap}s)")
