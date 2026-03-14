import os
from datetime import datetime

OUTPUT_FILE = "dataset_time_ranges.txt"

def extract_timestamp(line):
    parts = line.replace(",", " ").split()
    if not parts:
        return None
    try:
        return int(float(parts[0]))
    except ValueError:
        return None

def fmt(ts):
    return datetime.utcfromtimestamp(ts).strftime("%d-%m-%Y %H:%M:%S")

with open(OUTPUT_FILE, "w") as out:
    out.write(
        f"{'filename':<30} | "
        f"{'first_ts':<12} | "
        f"{'first_datetime_utc':<19} | "
        f"{'last_ts':<12} | "
        f"{'last_datetime_utc':<19}\n"
    )
    out.write("-" * 110 + "\n")


    for fname in sorted(os.listdir(".")):
        if not fname.startswith("feed") or not fname.endswith(".csv"):
            continue

        first_ts = None
        last_ts = None

        with open(fname, "r") as f:
            for line in f:
                ts = extract_timestamp(line)
                if ts is None:
                    continue
                if first_ts is None:
                    first_ts = ts
                last_ts = ts

        if first_ts is None:
            continue

        out.write(
            f"{fname:<30} | "
            f"{first_ts:<12} | "
            f"{fmt(first_ts):<19} | "
            f"{last_ts:<12} | "
            f"{fmt(last_ts):<19}\n"
        )


print(f"Saved timestamp ranges to {OUTPUT_FILE}")
