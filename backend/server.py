#!/usr/bin/env python3

import asyncio
import json
import pickle
from collections import deque
from pathlib import Path

import numpy as np
import socketio
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from live_simulator import LiveWaterFlowGenerator
from model import HybridWaterAnomalyDetector


# --------------------------------------------------
# Resolve Paths
# --------------------------------------------------

BASE_DIR      = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
FRONTEND_DIR  = BASE_DIR / "frontend"


# --------------------------------------------------
# Socket.IO + FastAPI Setup
# --------------------------------------------------

sio        = socketio.AsyncServer(cors_allowed_origins="*", async_mode="asgi")
app        = FastAPI()
socket_app = socketio.ASGIApp(sio, app)


# --------------------------------------------------
# Serve Frontend
# --------------------------------------------------

app.mount(
    "/static",
    StaticFiles(directory=FRONTEND_DIR / "static"),
    name="static",
)

@app.get("/")
async def serve_index():
    return FileResponse(FRONTEND_DIR / "index.html")


# --------------------------------------------------
# Load Artifacts
# --------------------------------------------------

with open(ARTIFACTS_DIR / "if_model.pkl", "rb") as f:
    if_model = pickle.load(f)

with open(ARTIFACTS_DIR / "if_scaler.pkl", "rb") as f:
    if_scaler = pickle.load(f)

with open(ARTIFACTS_DIR / "if_calibration.json") as f:
    cal = json.load(f)

WINDOW_MINUTES = cal["window_minutes"]

generator = LiveWaterFlowGenerator(BASE_DIR / "all_appliances.json")

detector = HybridWaterAnomalyDetector(
    if_model               = if_model,
    if_scaler              = if_scaler,
    cusum_k                = cal["cusum_k"],
    cusum_h                = cal["cusum_h"],
    if_threshold           = cal["if_threshold"],
    if_score_scale         = cal["if_score_scale"],
    appliance_flow_thresh  = cal["appliance_flow_thresh"],
)


# --------------------------------------------------
# Simulation State
# --------------------------------------------------

simulation_running = False
simulation_speed   = 1.0
sim_minutes        = 0

window_buffer: deque = deque(maxlen=WINDOW_MINUTES)

last_result = {
    "anomaly": False,
    "final_score": 0.0,
    "level2": {"triggered": False, "score": 0.0},
    "level3": {"triggered": False, "score": 0.0, "reconstruction_error": 0.0},
}

leak_active       = False
leak_intensity    = 0.0
leak_end_minute   = None
leak_start_minute = None
leak_mode         = "instant"
leak_ramp_minutes = 5


# --------------------------------------------------
# Background Simulation Loop
# --------------------------------------------------

async def simulation_loop():

    global simulation_running, simulation_speed, sim_minutes
    global leak_active, leak_intensity
    global leak_end_minute, leak_start_minute
    global leak_mode, leak_ramp_minutes
    global last_result

    while True:

        if simulation_running:

            flow = generator.next()

            # --------------------------------------------------
            # Apply Leak
            # --------------------------------------------------

            if leak_active:

                if leak_end_minute is not None and sim_minutes >= leak_end_minute:

                    leak_active       = False
                    leak_end_minute   = None
                    leak_start_minute = None

                else:

                    if leak_mode == "instant":

                        effective_intensity = leak_intensity

                    elif leak_mode == "ramp":

                        elapsed  = sim_minutes - leak_start_minute
                        progress = min(1.0, elapsed / max(1, leak_ramp_minutes))

                        effective_intensity = leak_intensity * progress

                    else:

                        effective_intensity = leak_intensity

                    flow += effective_intensity
                    flow  = min(flow, generator.MAX_FLOW_LPM)

            # --------------------------------------------------
            # Buffer window
            # --------------------------------------------------

            window_buffer.append(float(flow))

            if len(window_buffer) == WINDOW_MINUTES:

                last_result = detector.update(list(window_buffer))

            result = dict(last_result)
            result["flow"] = float(flow)

            # --------------------------------------------------
            # Simulated time
            # --------------------------------------------------

            sim_minutes += 1

            result["sim_time"] = f"{(sim_minutes // 60) % 24:02d}:{sim_minutes % 60:02d}"
            result["sim_minutes"] = sim_minutes

            # --------------------------------------------------
            # Leak metadata
            # --------------------------------------------------

            result["leak_active"]    = bool(leak_active)
            result["leak_mode"]      = str(leak_mode)
            result["leak_intensity"] = float(leak_intensity if leak_active else 0.0)

            result["leak_remaining"] = (
                max(0, leak_end_minute - sim_minutes)
                if leak_active and leak_end_minute
                else 0
            )

            # --------------------------------------------------
            # Emit
            # --------------------------------------------------

            await sio.emit("data_update", result)

            await asyncio.sleep(1.0 / simulation_speed)

        else:

            await asyncio.sleep(0.1)


@app.on_event("startup")
async def startup_event():

    asyncio.create_task(simulation_loop())


# --------------------------------------------------
# Simulation Controls
# --------------------------------------------------

@sio.event
async def start_simulation(sid):

    global simulation_running

    simulation_running = True

    await sio.emit("simulation_state", {"state": "running"})


@sio.event
async def pause_simulation(sid):

    global simulation_running

    simulation_running = False

    await sio.emit("simulation_state", {"state": "paused"})


@sio.event
async def stop_simulation(sid):

    global simulation_running, sim_minutes
    global leak_active, leak_end_minute, leak_start_minute

    simulation_running = False
    sim_minutes = 0

    leak_active       = False
    leak_end_minute   = None
    leak_start_minute = None

    window_buffer.clear()

    generator.current_day    = 0
    generator.current_minute = 0
    generator._generate_new_day()

    detector.reset()

    await sio.emit("simulation_state", {"state": "stopped"})


@sio.event
async def set_speed(sid, data):

    global simulation_speed

    try:

        simulation_speed = max(1.0, min(float(data), 10.0))

        await sio.emit("speed_update", {"speed": simulation_speed})

    except Exception:

        pass


# --------------------------------------------------
# Leak Controls
# --------------------------------------------------

@sio.event
async def inject_leak(sid, data):

    global leak_active, leak_intensity
    global leak_end_minute, leak_start_minute
    global leak_mode, leak_ramp_minutes

    try:

        intensity    = max(0.1, min(float(data.get("intensity", 0.5)), 2.0))
        duration     = max(1, int(data.get("duration", 60)))
        mode         = data.get("mode", "instant")
        ramp_minutes = max(1, int(data.get("ramp_minutes", 5)))

        leak_active       = True
        leak_intensity    = intensity
        leak_end_minute   = sim_minutes + duration
        leak_start_minute = sim_minutes
        leak_mode         = mode if mode in ["instant", "ramp"] else "instant"
        leak_ramp_minutes = ramp_minutes

        await sio.emit("leak_status", {
            "active": True,
            "mode": leak_mode,
            "intensity": leak_intensity
        })

    except Exception:

        pass


@sio.event
async def stop_leak(sid):

    global leak_active, leak_end_minute, leak_start_minute

    leak_active       = False
    leak_end_minute   = None
    leak_start_minute = None

    await sio.emit("leak_status", {"active": False})


# --------------------------------------------------
# Run Server
# --------------------------------------------------

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(socket_app, host="0.0.0.0", port=8000, log_level="info")