import json
import matplotlib.pyplot as plt

PRIORS_PATH = "all_appliances.json"

with open(PRIORS_PATH) as f:
    priors = json.load(f)["appliances"]

# Proper display names
NAME_MAP = {
    "shower": "Shower",
    "washingmachine": "Washing Machine",
    "bidet": "Bidet",
    "washbasin": "Wash Basin",
    "kitchenfaucet": "Kitchen Faucet",
    "toilet": "Toilet"
}

plt.style.use("seaborn-v0_8-whitegrid")

fig, axes = plt.subplots(3, 2, figsize=(10,8), sharex=True, sharey=True)

axes = axes.flatten()

for i, ap in enumerate(priors):

    probs = ap["timing"]["start_hour"]["p"]
    name = NAME_MAP.get(ap["appliance"], ap["appliance"].title())

    ax = axes[i]

    ax.plot(range(24), probs, marker="o")

    ax.set_title(name)
    ax.set_xlim(0,23)

    ax.set_xlabel("Hour")
    ax.set_ylabel("P")

# fig.suptitle("Appliance Activation Probability by Hour", fontsize=16)

plt.tight_layout()
plt.show()