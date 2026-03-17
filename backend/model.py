"""
Hybrid Water Anomaly Detection Model

DESCRIPTION:
    Implements a hybrid anomaly detector for household water flow data, combining
    statistical CUSUM change detection with a machine learning Isolation Forest.
    Used for real-time leak/anomaly detection in streaming or batch settings.

KEY FEATURES:
    - CUSUM (level 2): Detects persistent low-flow deviations (leaks)
    - Isolation Forest (level 3): Detects statistical anomalies in feature space
    - Feature extraction: 5 features per window (see _extract_features)
    - Persistence filter: Reduces false positives by requiring consecutive anomalies
    - Tunable thresholds and weights for flexible deployment

DEPENDENCIES:
    - numpy: Data processing
    - sklearn: Model/scaler (passed in from training script)
"""

import numpy as np


class HybridWaterAnomalyDetector:

    def __init__(
        self,
        if_model,
        if_scaler,
        cusum_k=0.01,
        cusum_h=2.0,
        noise_floor=0.02,
        if_threshold=-0.05,
        if_score_scale=0.1,
        appliance_flow_thresh=0.8,  # lowered
        clip_bound=10.0,
        w2=0.4,
        w3=0.6,
        decision_threshold=0.65,
        persistence_windows=2,      # NEW
    ):
        """
        Initialize the hybrid anomaly detector with model, scaler, and detection parameters.
        """

        self.if_model = if_model
        self.if_scaler = if_scaler

        self.cusum_k = cusum_k
        self.cusum_h = cusum_h
        self.noise_floor = noise_floor

        self.if_threshold = if_threshold
        self.if_score_scale = if_score_scale
        self.clip_bound = clip_bound

        self.appliance_flow_thresh = appliance_flow_thresh

        self.w2 = w2
        self.w3 = w3
        self.decision_threshold = decision_threshold

        self.persistence_windows = persistence_windows
        self._anomaly_streak = 0

        self.cusum_s = 0.0
        self._prev_appliance = False


    def _run_cusum(self, window):
        """
        Run CUSUM change detection on a window of flow data.
        Returns the final CUSUM statistic and whether a change was triggered.
        """

        s = self.cusum_s
        triggered = False
        prev = self._prev_appliance

        for lpm in window:

            appliance = lpm >= self.appliance_flow_thresh

            if appliance and not prev:
                s = 0.0

            if not appliance:

                if lpm <= self.noise_floor:
                    s *= 0.8
                else:
                    delta = lpm - self.cusum_k
                    s = max(0.0, s + delta)

                if s >= self.cusum_h:
                    triggered = True

            prev = appliance

        self.cusum_s = s
        self._prev_appliance = prev

        return s, triggered


    def _extract_features(self, window):
        """
        Extract 5 statistical features from a window for anomaly detection.
        Features: mnf, inter_mean, inter_frac, mean_flow, inter_std.
        Returns a scaled and clipped feature vector for the ML model.
        """

        inter = window[window < self.appliance_flow_thresh]
        nonzero = window[window > 0.0]

        mnf = float(np.percentile(nonzero, 10)) if len(nonzero) > 0 else 0.0

        inter_mean = float(inter.mean()) if len(inter) > 0 else 0.0
        inter_frac = float((inter > self.noise_floor).mean()) if len(inter) > 0 else 0.0
        inter_std = float(inter.std()) if len(inter) > 0 else 0.0

        mean_flow = float(window.mean())

        raw = np.array(
            [[mnf, inter_mean, inter_frac, mean_flow, inter_std]],
            dtype=np.float32,
        )

        scaled = self.if_scaler.transform(raw)
        clipped = np.clip(scaled, -self.clip_bound, self.clip_bound)

        return clipped


    def update(self, window):
        """
        Update the detector with a new window of flow data.
        Combines CUSUM and Isolation Forest scores, applies persistence filter.
        Returns a dictionary with anomaly status and detailed scores.
        """

        window = np.asarray(window, dtype=np.float32)

        s_final, cusum_triggered = self._run_cusum(window)
        cusum_score = min(1.0, s_final / self.cusum_h)

        features = self._extract_features(window)

        raw_if_score = float(self.if_model.decision_function(features)[0])

        if_triggered = raw_if_score < self.if_threshold

        anomaly_distance = max(0.0, self.if_threshold - raw_if_score)
        if_score = min(1.0, anomaly_distance / self.if_score_scale)

        final_score = self.w2 * cusum_score + self.w3 * if_score

        candidate_anomaly = final_score > self.decision_threshold

        # ── persistence filter ──
        if candidate_anomaly:
            self._anomaly_streak += 1
        else:
            self._anomaly_streak = 0

        final_anomaly = self._anomaly_streak >= self.persistence_windows

        return {
            "anomaly": bool(final_anomaly),
            "final_score": float(final_score),
            "level2": {
                "triggered": bool(cusum_triggered),
                "score": float(cusum_score),
            },
            "level3": {
                "triggered": bool(if_triggered),
                "score": float(if_score),
                "reconstruction_error": float(raw_if_score),
            },
        }


    def reset(self):
        """
        Reset the detector's internal state (CUSUM, persistence streak).
        """

        self.cusum_s = 0.0
        self._prev_appliance = False
        self._anomaly_streak = 0