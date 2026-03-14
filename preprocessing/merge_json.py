import json
import os
from pathlib import Path

# -----------------------------
# CONFIG
# -----------------------------
INPUT_DIR = "./priors_india"          # directory containing individual JSON files
OUTPUT_FILE = "all_appliances.json"

# -----------------------------
# LOAD + MERGE
# -----------------------------
all_appliances = []

for file in Path(INPUT_DIR).glob("*.json"):
    with open(file, "r") as f:
        data = json.load(f)

        # Case 1: single appliance object
        if isinstance(data, dict):
            all_appliances.append(data)

        # Case 2: list of appliance objects
        elif isinstance(data, list):
            all_appliances.extend(data)

        else:
            raise ValueError(f"Unsupported JSON structure in {file}")

# -----------------------------
# FINAL WRAP
# -----------------------------
output = {
    "schema_version": "appliance_priors_v1",
    "count": len(all_appliances),
    "appliances": all_appliances
}

# -----------------------------
# WRITE OUTPUT
# -----------------------------
with open(OUTPUT_FILE, "w") as f:
    json.dump(output, f, indent=2)

print(f"✅ Merged {len(all_appliances)} appliances into {OUTPUT_FILE}")
