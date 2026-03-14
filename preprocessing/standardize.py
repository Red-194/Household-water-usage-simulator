import os

RAW_DIR = "data"
OUT_DIR = "processed"

os.makedirs(OUT_DIR, exist_ok=True)

def parse(line):
    return line.replace(",", " ").split()

for fname in sorted(os.listdir(RAW_DIR)):
    if not fname.startswith("feed") or not fname.endswith(".csv"):
        continue

    base = (
        fname.replace("feed_", "")
             .replace("feed.", "")
             .replace(".MYD", "")
             .replace(".csv", "")
             .lower()
    )

    point_out = open(os.path.join(OUT_DIR, f"{base}_points.csv"), "w")
    interval_out = open(os.path.join(OUT_DIR, f"{base}_intervals.csv"), "w")

    point_out.write("timestamp,flow_ml_s\n")
    interval_out.write("start_ts,end_ts,flow_ml_s\n")

    with open(os.path.join(RAW_DIR, fname)) as f:
        for line in f:
            if any(c.isalpha() for c in line):
                continue  # skip header

            parts = parse(line)

            if len(parts) == 2:
                ts, flow = parts
                point_out.write(f"{ts},{flow}\n")

            elif len(parts) == 3:
                start, flow, end = parts
                interval_out.write(f"{start},{end},{flow}\n")

    point_out.close()
    interval_out.close()

    print(f"Processed {fname}")
