"""
Live Water Flow Simulation for Real-Time Applications

DESCRIPTION:
    Provides a generator class for simulating minute-by-minute household water flow
    in a live/online fashion. Used for real-time anomaly/leak detection demos and
    interactive dashboards. Supports leak injection, sensor noise, and appliance-aware
    event generation, closely mirroring the batch simulator but with stateful streaming.

KEY FEATURES:
    - Realistic appliance event generation (same logic as batch simulator)
    - Daily volume constraints and multi-attempt regeneration
    - Additive sensor noise on non-zero flows
    - Leak injection (adds flow for a configurable duration/intensity)
    - Stateless API: call next() to get the next minute's flow

DEPENDENCIES:
    - numpy, json: Data processing and appliance config
"""

import json
import numpy as np
import warnings

MINUTES_PER_DAY = 1440
SECONDS_PER_MIN = 60
DEFAULT_DAILY_MIN_L = 100
DEFAULT_DAILY_MAX_L = 160


class LiveWaterFlowGenerator:
    def __init__(
        self,
        priors_path,
        daily_min_l=DEFAULT_DAILY_MIN_L,
        daily_max_l=DEFAULT_DAILY_MAX_L,
        max_flow_lpm=15.0,
        noise_sigma=0.03,
        max_regen_attempts=10,
        seed=None,
    ):
        """
        Initialize the live water flow generator.
        Loads appliance priors, sets simulation parameters, and generates the first day.
        """
        if seed is not None:
            np.random.seed(seed)

        self.wm_offset = np.random.randint(7)

        with open(priors_path) as f:
            self.priors = json.load(f)["appliances"]

        self.DAILY_MIN_L = daily_min_l
        self.DAILY_MAX_L = daily_max_l
        self.MAX_FLOW_LPM = max_flow_lpm
        self.NOISE_SIGMA = noise_sigma
        self.MAX_REGEN_ATTEMPTS = max_regen_attempts

        self.current_day = 0
        self.current_minute = 0
        self.day_flow = None

        self.injected_leak = None

        self._generate_new_day()

    def next(self):
        """
        Return the next minute's flow value (LPM), including sensor noise and any injected leak.
        Advances the simulation by one minute, generating a new day if needed.
        """
        if self.current_minute >= MINUTES_PER_DAY:
            self.current_day += 1
            self.current_minute = 0
            self._generate_new_day()

        flow_value = self.day_flow[self.current_minute]  # type: ignore

        # Leak adds to existing flow instead of replacing it
        if self.injected_leak is not None:
            global_min = self.global_minute()
            if (
                self.injected_leak["start"]
                <= global_min
                < self.injected_leak["end"]
            ):
                flow_value += self.injected_leak["flow_lpm"]

        # Additive sensor noise — only when flow is non-zero
        if flow_value > 0:
            noise = np.random.normal(0.0, self.NOISE_SIGMA * flow_value)
            flow_value = max(0.0, flow_value + noise)

        self.current_minute += 1
        return float(flow_value)

    def inject_leak(self, duration_minutes=180, flow_lpm=0.4):
        """
        Inject a synthetic leak event for a given duration and flow rate (LPM).
        Leak is additive to normal appliance flow.
        """
        start = self.global_minute()
        end = start + duration_minutes
        self.injected_leak = {
            "start": start,
            "end": end,
            "flow_lpm": flow_lpm,
        }

    def clear_leak(self):
        """
        Remove any active leak from the simulation.
        """
        self.injected_leak = None

    def global_minute(self):
        """
        Return the current global minute index (across all simulated days).
        """
        return self.current_day * MINUTES_PER_DAY + self.current_minute

    def _generate_new_day(self):
        """
        Generate a new day's worth of appliance events and flow, retrying if daily volume is out of bounds.
        """
        for attempt in range(self.MAX_REGEN_ATTEMPTS):
            day_events = []
            for appliance in self.priors:
                day_events.extend(
                    self._generate_events_for_day(appliance, self.current_day)
                )

            day_flow = self._render_day(day_events, self.current_day)
            day_volume = day_flow.sum()

            if self.DAILY_MIN_L <= day_volume <= self.DAILY_MAX_L:
                break

            if attempt == self.MAX_REGEN_ATTEMPTS - 1:
                warnings.warn(
                    f"Day {self.current_day}: volume {day_volume:.1f} L outside "
                    f"[{self.DAILY_MIN_L}, {self.DAILY_MAX_L}] after "
                    f"{self.MAX_REGEN_ATTEMPTS} attempts. Accepting anyway."
                )

        self.day_flow = day_flow    # type: ignore

    def _generate_events_for_day(self, appliance, day):
        """
        Generate events for a single appliance on a given day, using Poisson sampling and appliance constraints.
        """
        name       = appliance["appliance"]
        lam        = appliance["activation"]["events_per_day"]["lambda"]
        hour_probs = appliance["timing"]["start_hour"]["p"]

        if name == "shower":
            n_events = max(1, np.random.poisson(lam))
        elif name == "toilet":
            n_events = max(2, np.random.poisson(lam))
        elif name == "bidet":
            n_events = np.random.poisson(lam)
        elif name == "washingmachine":
            n_events = 1 if (day + self.wm_offset) % 7 == 0 else 0
        else:
            n_events = np.random.poisson(lam)

        events = []

        for _ in range(n_events):
            start_min = day * MINUTES_PER_DAY + (
                np.random.choice(24, p=hour_probs) * 60 + np.random.randint(60)
            )

            dur_cfg = appliance["duration"]

            if dur_cfg["type"] == "fixed":
                duration_s = dur_cfg["value"]
            elif name in ["washbasin", "kitchenfaucet", "bidet", "toilet"]:
                duration_s = np.clip(
                    np.random.normal(dur_cfg["scale"], 0.25 * dur_cfg["scale"]),
                    appliance["constraints"]["min_duration_s"],
                    appliance["constraints"]["max_duration_s"],
                )
            else:
                duration_s = np.random.lognormal(
                    mean=np.log(dur_cfg["scale"]),
                    sigma=dur_cfg["shape"],
                )

            duration_s = np.clip(
                duration_s,
                appliance["constraints"]["min_duration_s"],
                appliance["constraints"]["max_duration_s"],
            )

            flow_cfg = appliance["flow"]["mean_flow"]
            mean_flow_ml_s = np.random.lognormal(
                mean=np.log(flow_cfg["scale"]),
                sigma=flow_cfg["shape"],
            )

            events.append({
                "start_min":      int(start_min),
                "duration_s":     float(duration_s),
                "mean_flow_ml_s": float(mean_flow_ml_s),
                "shape":          appliance.get("shape", {}).get("type", "step"),
                "shape_cfg":      appliance.get("shape", {}),
            })

        return events

    def _render_day(self, events, day):
        """
        Render a day's worth of flow from a list of events, applying shape curves and flow limits.
        """
        flow = np.zeros(MINUTES_PER_DAY)

        for ev in events:
            start      = ev["start_min"] - day * MINUTES_PER_DAY
            dur        = int(np.ceil(ev["duration_s"] / SECONDS_PER_MIN))
            lpm        = ev["mean_flow_ml_s"] * 60 / 1000
            end        = min(start + dur, MINUTES_PER_DAY)
            actual_dur = end - start

            if actual_dur <= 0:
                continue

            curve = self._make_shape_curve(ev["shape"], ev["shape_cfg"], actual_dur)

            flow[start:end] += lpm * curve
            flow[start:end]  = np.minimum(flow[start:end], self.MAX_FLOW_LPM)

        return flow

    def _make_shape_curve(self, shape, shape_cfg, dur):
        """
        Generate a normalized flow shape curve for an event (trapezoid, pulsed, or step).
        """
        if shape == "trapezoid" and dur >= 4:
            ramp_s = shape_cfg.get("ramp_up_s", 5)
            fall_s = shape_cfg.get("ramp_down_s", 5)

            ramp_bins    = max(1, int(np.ceil(ramp_s / SECONDS_PER_MIN)))
            fall_bins    = max(1, int(np.ceil(fall_s / SECONDS_PER_MIN)))
            plateau_bins = max(0, dur - ramp_bins - fall_bins)

            if plateau_bins <= 0:
                return np.ones(dur)

            curve = np.concatenate([
                np.linspace(0.5, 1.0, ramp_bins),
                np.ones(plateau_bins),
                np.linspace(1.0, 0.5, fall_bins),
            ])

        elif shape == "pulsed" and dur >= 2:
            pulse = (np.arange(dur) % 4 < 2).astype(float)
            if pulse.sum() == 0:
                return np.ones(dur)
            curve = pulse

        else:
            return np.ones(dur)

        mean = curve.mean()
        return curve / mean if mean > 0 else np.ones(dur)