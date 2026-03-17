# Household Water Usage Simulator

A comprehensive, research-grade Python framework for generating synthetic household water consumption data and detecting anomalies (leaks) using hybrid statistical and machine learning methods.

## Overview

This project simulates realistic minute-by-minute water usage patterns for Indian households based on **BIS IS 1172 standards** (100-160 L/day). It combines:

- **Realistic simulation** of appliance-specific water flow events (showers, toilets, washing machines, etc.)
- **Hybrid anomaly detection** using CUSUM (statistical) and Isolation Forest (ML) methods
- **Real-time streaming dashboard** via FastAPI and Socket.IO for monitoring and leak injection
- **Synthetic dataset generation** for ML model training and validation

## Features

✅ Minute-level flow rate simulation (365-day datasets)  
✅ Appliance-aware event generation with realistic flow curves  
✅ Multi-method anomaly detection with persistence filtering  
✅ Real-time web dashboard with live simulation and leak injection  
✅ Configurable leaks (instant or ramp-up modes)  
✅ Reproducible, deterministic synthetic data  
✅ Production-ready documentation and type safety  

---

## Project Structure

```
Household-water-usage-simulator/
├── README.md                          # This file
├── all_appliances.json                # Appliance configuration (flow profiles, durations)
│
├── backend/
│   ├── core_simulator.py              # Core water flow simulation engine
│   ├── live_simulator.py              # Real-time simulation wrapper for streaming
│   ├── isolation_forest.py            # Isolation Forest model wrapper
│   ├── model.py                       # Hybrid anomaly detector (CUSUM + IF)
│   └── server.py                      # FastAPI + Socket.IO server
│
├── frontend/
│   ├── index.html                     # Main dashboard page
│   ├── static/
│   │   ├── css/
│   │   ├── js/
│   │   └── assets/
│   └── ...
│
├── artifacts/
│   ├── if_model.pkl                   # Trained Isolation Forest model
│   ├── if_scaler.pkl                  # Feature scaler (StandardScaler)
│   └── if_calibration.json            # Detection thresholds and weights
│
└── simulator_data/
    ├── household_flow_365d.csv        # Generated flow time series (1440 × 365 rows)
    └── household_events_365d.csv      # Event log with appliance, timing, duration
```

---

## Installation

### Prerequisites

- **Python 3.8+**
- **pip** or **conda**

### Setup

1. **Clone or download the project:**
   ```bash
   cd /home/adithya-manghat/Desktop/Household-water-usage-simulator
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   **Required packages:**
   - `numpy` — Numerical computations
   - `pandas` — Data manipulation and CSV I/O
   - `scikit-learn` — Isolation Forest, scalers
   - `fastapi` — Web framework
   - `python-socketio` — Real-time WebSocket communication
   - `uvicorn` — ASGI server
   - `matplotlib` (optional) — Plotting results

---

## Usage

### 1. Generate Synthetic Data (Offline)

To generate a full 365-day synthetic dataset:

```bash
cd backend
python core_simulator.py
```

**Output:**
- `simulator_data/household_flow_365d.csv` — Minute-level flow rates
- `simulator_data/household_events_365d.csv` — Event log

### 2. Run the Real-Time Server

Start the FastAPI + Socket.IO server:

```bash
cd backend
python server.py
```

**Expected output:**
```
INFO:     Started server process [PID]
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 3. Access the Dashboard

Open your browser and navigate to:
```
http://localhost:8000
```

You should see a real-time dashboard with:
- Live water flow visualization
- Anomaly detection scores and status
- Leak injection controls
- Simulation speed and pause/resume buttons

### 4. Interact with the Simulator

**Controls available in the dashboard:**

| Action | Effect |
|--------|--------|
| **Start** | Begin simulation |
| **Pause** | Pause simulation (can resume) |
| **Stop** | Reset simulation to day 0, minute 0 |
| **Speed** | Adjust simulation speed (1x–10x) |
| **Inject Leak** | Add synthetic leak (instant or ramp mode) |
| **Stop Leak** | End active leak |

---

## API Documentation

### Socket.IO Events (Frontend → Server)

| Event | Payload | Purpose |
|-------|---------|---------|
| `start_simulation` | — | Start/resume simulation |
| `pause_simulation` | — | Pause simulation |
| `stop_simulation` | — | Reset and stop |
| `set_speed` | `{"speed": float}` | Set simulation speed (1–10x) |
| `inject_leak` | `{"intensity": float, "duration": int, "mode": str, "ramp_minutes": int}` | Inject a leak |
| `stop_leak` | — | Stop active leak |

### Socket.IO Events (Server → Frontend)

| Event | Payload | Purpose |
|-------|---------|---------|
| `data_update` | Flow, anomaly scores, leak metadata | Real-time data update |
| `simulation_state` | `{"state": str}` | Simulation state change |
| `speed_update` | `{"speed": float}` | Speed update confirmation |
| `leak_status` | `{"active": bool, ...}` | Leak status update |

---

## Configuration

### Simulation Parameters

Edit values in `backend/core_simulator.py`:

```python
DAYS = 365                    # Number of days to simulate
MINUTES_PER_DAY = 1440        # Minutes per day
DAILY_MIN_L = 100             # Minimum daily consumption (liters)
DAILY_MAX_L = 160             # Maximum daily consumption (liters)
MAX_FLOW_LPM = 15.0           # Maximum flow rate (liters/minute)
NOISE_SIGMA = 0.03            # Sensor noise standard deviation
```

### Anomaly Detection Thresholds

Edit values in `backend/artifacts/if_calibration.json`:

```json
{
  "window_minutes": 10,
  "cusum_k": 0.5,
  "cusum_h": 5.0,
  "if_threshold": -0.3,
  "if_score_scale": 1.0,
  "decision_threshold": 0.5,
  "persistence_windows": 2
}
```

**Parameters:**
- `window_minutes` — Detection window size (minutes)
- `cusum_k` — CUSUM reference value (slack parameter)
- `cusum_h` — CUSUM decision threshold
- `if_threshold` — Isolation Forest decision boundary
- `decision_threshold` — Final anomaly decision threshold
- `persistence_windows` — Minimum consecutive anomalous windows to flag

---

## Examples

### Example 1: Generate and Inspect Synthetic Data

```python
from backend.core_simulator import HouseholdWaterSimulator

sim = HouseholdWaterSimulator()
sim.generate_365_day_dataset()

# Load and inspect
import pandas as pd
flow_df = pd.read_csv("simulator_data/household_flow_365d.csv")
print(flow_df.describe())
```

### Example 2: Run Anomaly Detection on Synthetic Data

```python
import pickle
import numpy as np
from backend.model import HybridWaterAnomalyDetector

# Load model and scaler
with open("artifacts/if_model.pkl", "rb") as f:
    if_model = pickle.load(f)
with open("artifacts/if_scaler.pkl", "rb") as f:
    if_scaler = pickle.load(f)

# Initialize detector
detector = HybridWaterAnomalyDetector(if_model=if_model, if_scaler=if_scaler)

# Test on a window of flow data
test_window = np.random.normal(2.0, 0.5, 10)  # Example data
result = detector.update(test_window)
print(f"Anomaly detected: {result['anomaly']}")
print(f"Score: {result['final_score']:.3f}")
```

### Example 3: Inject a Leak via API

Once the server is running, send a leak injection request:

```python
import socketio

sio = socketio.Client()
sio.connect("http://localhost:8000")

sio.emit("inject_leak", {
    "intensity": 1.0,          # L/min added
    "duration": 120,           # 2 hours
    "mode": "ramp",            # Gradually increase
    "ramp_minutes": 10         # Ramp over 10 minutes
})

sio.wait()
```

---

## Research & Publications

This simulator is designed for research in:

- **Water anomaly detection** (leak detection)
- **Time series analysis** with hybrid ML/statistical methods
- **Smart water meter datasets** for IoT and smart home systems
- **Benchmark datasets** for anomaly detection algorithms

If you use this in your research, please cite it as:

```
Household Water Usage Simulator. (2026). 
A comprehensive framework for synthetic water consumption data generation 
and anomaly detection. GitHub repository.
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'fastapi'` | Run `pip install -r requirements.txt` |
| `FileNotFoundError: all_appliances.json` | Ensure working directory is `backend/` |
| `Connection refused on localhost:8000` | Check that the server is running; try a different port |
| Dashboard not loading | Clear browser cache and refresh; check browser console for errors |

---

## Contributing

For bug reports, feature requests, or contributions, please create an issue or pull request.

---

## License

This project is provided as-is for research and educational purposes.

---

## Contact & Support

For questions or support, reach out to the project maintainers or open an issue on the repository.

---

**Last Updated:** March 2026