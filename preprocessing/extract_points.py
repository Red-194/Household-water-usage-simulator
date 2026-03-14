import csv
import os

filename = "washingmachine"

INPUT_FILE = f"processed/{filename}_points.csv"   # change per fixture
OUTPUT_FILE = f"events/{filename}_events.csv"

os.makedirs("events", exist_ok=True)

events = []
event_id = 0

current_start = None
current_end = None
flows = []

with open(INPUT_FILE) as f:
    reader = csv.DictReader(f)
    prev_ts = None

    for row in reader:
        ts = int(row["timestamp"])
        flow = float(row["flow_ml_s"])

        if flow > 0:
            if current_start is None:
                # start new event
                current_start = ts
                flows = [flow]
            else:
                # check continuity
                if prev_ts is not None and ts - prev_ts > 1:
                    # gap → close previous event
                    duration = current_end - current_start + 1
                    events.append((
                        event_id,
                        current_start,
                        current_end,
                        duration,
                        sum(flows) / len(flows),
                        max(flows),
                        sum(flows)  # since 1 Hz, ml/s * s
                    ))
                    event_id += 1
                    current_start = ts
                    flows = [flow]
                else:
                    flows.append(flow)

            current_end = ts

        else:
            if current_start is not None:
                duration = current_end - current_start + 1
                events.append((
                    event_id,
                    current_start,
                    current_end,
                    duration,
                    sum(flows) / len(flows),
                    max(flows),
                    sum(flows)
                ))
                event_id += 1
                current_start = None
                flows = []

        prev_ts = ts

# close last event if open
if current_start is not None:
    duration = current_end - current_start + 1
    events.append((
        event_id,
        current_start,
        current_end,
        duration,
        sum(flows) / len(flows),
        max(flows),
        sum(flows)
    ))

# write output
with open(OUTPUT_FILE, "w", newline="") as f:
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
    writer.writerows(events)

print(f"Extracted {len(events)} events → {OUTPUT_FILE}")
