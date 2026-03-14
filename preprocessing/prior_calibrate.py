import json
from pathlib import Path
import copy

# -------------------------
# Paths
# -------------------------
PRIORS_IN_DIR = Path("priors")
PRIORS_OUT_DIR = Path("priors_india")
PRIORS_OUT_DIR.mkdir(exist_ok=True)

# -------------------------
# Flow calibration factors
# -------------------------
FLOW_SCALE = {
    "kitchenfaucet": {"mean": 3.0, "peak": 4.0},
    "washbasin":     {"mean": 2.5, "peak": 3.5},
    "shower":        {"mean": 2.0, "peak": 2.5},
    "bidet":         {"mean": 3.0, "peak": 4.0},
    "toilet":        {"mean": 1.0, "peak": 1.0},
    "dishwasher30":  {"mean": 1.0, "peak": 1.0},
    "washingmachine":{"mean": 1.2, "peak": 1.5},
}

# -------------------------
# Helpers
# -------------------------
def scale_lognormal(dist, factor):
    """
    For lognormal:
      mean = scale * exp(0.5 * shape^2)
    Scaling the variable by k -> scale *= k
    """
    if dist["type"] != "lognormal":
        return dist

    out = dist.copy()
    out["scale"] = out["scale"] * factor
    return out


def scale_fixed(dist, factor):
    if dist["type"] != "fixed":
        return dist

    out = dist.copy()
    out["value"] = out["value"] * factor
    return out


def scale_distribution(dist, factor):
    if dist["type"] == "lognormal":
        return scale_lognormal(dist, factor)
    elif dist["type"] == "fixed":
        return scale_fixed(dist, factor)
    else:
        return dist


# -------------------------
# Calibration
# -------------------------
def calibrate_prior(prior):
    appliance = prior["appliance"]

    if appliance not in FLOW_SCALE:
        print(f"[SKIP] No calibration rule for {appliance}")
        return prior

    factors = FLOW_SCALE[appliance]
    calibrated = copy.deepcopy(prior)

    # ---- scale flows ----
    calibrated["flow"]["mean_flow"] = scale_distribution(
        calibrated["flow"]["mean_flow"],
        factors["mean"]
    )

    calibrated["flow"]["peak_flow"] = scale_distribution(
        calibrated["flow"]["peak_flow"],
        factors["peak"]
    )

    # ---- annotate calibration ----
    calibrated["calibration"] = {
        "region": "India",
        "method": "deterministic scaling",
        "flow_scale_factors": factors,
        "notes": (
            "Flow rates scaled to reflect Indian domestic fixtures "
            "(MoHUA / BIS guidance, manufacturer specs). "
            "Timing, duration, and activation priors unchanged."
        )
    }

    calibrated["version"] = "1.1-india"

    return calibrated


# -------------------------
# Main
# -------------------------
def main():
    for prior_file in PRIORS_IN_DIR.glob("*.json"):
        with open(prior_file) as f:
            prior = json.load(f)

        calibrated = calibrate_prior(prior)

        out_path = PRIORS_OUT_DIR / prior_file.name
        with open(out_path, "w") as f:
            json.dump(calibrated, f, indent=2)

        print(f"[OK] Calibrated {prior['appliance']} → {out_path.name}")


if __name__ == "__main__":
    main()
