import csv
import os

filename = "toilet"

INPUT_FILE = f"processed/{filename}_intervals.csv"
OUTPUT_FILE = f"events/{filename}_events.csv"

os.makedirs("events", exist_ok=True)

with open(INPUT_FILE) as f, open(OUTPUT_FILE, "w", newline="") as out:
    reader = csv.DictReader(f)
    writer = csv.writer(out)

    writer.writerow([
        "event_id",
        "start_ts",
        "end_ts",
        "duration_s",
        "mean_flow_ml_s",
        "peak_flow_ml_s",
        "total_volume_ml"
    ])

    for i, row in enumerate(reader):
        start = int(row["start_ts"])
        end = int(row["end_ts"])
        flow = float(row["flow_ml_s"])
        duration = end - start

        writer.writerow([
            i,
            start,
            end,
            duration,
            flow,
            flow,
            flow * duration
        ])

print(f"Extracted events → {OUTPUT_FILE}")
