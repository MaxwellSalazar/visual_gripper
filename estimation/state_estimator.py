"""
========================================================================
  estimation/state_estimator.py
  Stage B — Estimation: Proprioceptive State from Visual Observations
========================================================================
  Converts raw centroid positions (px, py) into:
    1. Deflection δ  [pixels]   — Euclidean distance from rest pose P₀
    2. Virtual Force F [Newtons] — via calibrated linear transfer function

  Scientific basis:
    δ = √( (px - P₀x)² + (py - P₀y)² )         [Euclidean metric]
    F = k_s · δ + b                               [linear regression fit]

  The rest pose P₀ is set once at startup (or loaded from file) and
  represents the undeformed state of the finger — the "zero-force" datum.
"""

import numpy as np
import json
import os
from dataclasses import dataclass, field
from typing import Optional, Tuple
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import (
    CALIB_SLOPE, CALIB_INTERCEPT,
    DEFLECTION_CONTACT, DEFLECTION_SOFT_STOP, DEFLECTION_HARD_STOP
)

# Path for persisted rest-pose reference
_REST_POSE_FILE = os.path.join(
    os.path.dirname(__file__), "..", "data", "rest_pose.json"
)


@dataclass
class GripperState:
    """Complete state snapshot at time t."""
    centroid: Tuple[float, float]       # (px, py) current marker position
    rest_pose: Tuple[float, float]      # (P₀x, P₀y) reference position
    deflection_px: float                # δ in pixels
    force_N: float                      # estimated force in Newtons
    contact: bool                       # True if δ > DEFLECTION_CONTACT
    regime: str                         # "free" | "contact" | "soft_stop" | "hard_stop"
    velocity_px_s: float = 0.0         # dδ/dt (filtered) in px/s


class StateEstimator:
    """
    Maintains the finger's proprioceptive state using visual observations.

    Usage
    -----
    est = StateEstimator()
    est.set_rest_pose((320.0, 240.0))   # call with finger at rest
    state = est.update((318.5, 252.3))  # call every frame
    print(state.force_N)
    """

    def __init__(
        self,
        calib_slope: float = CALIB_SLOPE,
        calib_intercept: float = CALIB_INTERCEPT,
        velocity_alpha: float = 0.7,   # EMA filter coefficient for velocity
    ):
        self._slope     = calib_slope
        self._intercept = calib_intercept
        self._alpha     = velocity_alpha   # higher → more filtering

        self._rest: Optional[Tuple[float, float]] = None
        self._prev_delta: float = 0.0
        self._prev_time: Optional[float] = None
        self._velocity_filtered: float = 0.0

        # Try to load persisted rest pose from last session
        self._try_load_rest_pose()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_rest_pose(self, centroid: Tuple[float, float]) -> None:
        """
        Record the current marker centroid as the zero-deflection datum.
        Call this with the gripper fully open and finger unloaded.
        Persists to disk so it survives restarts.
        """
        self._rest = centroid
        self._save_rest_pose(centroid)
        print(f"[StateEstimator] Rest pose set → P₀ = {centroid}")

    def update(
        self,
        centroid: Tuple[float, float],
        timestamp: Optional[float] = None,
    ) -> GripperState:
        """
        Compute the full gripper state from the current marker centroid.

        Parameters
        ----------
        centroid  : (px, py) from perception module
        timestamp : time.perf_counter() value (for velocity estimation)

        Returns
        -------
        GripperState
        """
        if self._rest is None:
            raise RuntimeError(
                "Rest pose not set. Call set_rest_pose() before update()."
            )

        px, py    = centroid
        p0x, p0y  = self._rest

        # ── Deflection ────────────────────────────────────────────────
        delta = float(np.sqrt((px - p0x) ** 2 + (py - p0y) ** 2))

        # ── Velocity dδ/dt (EMA filtered) ────────────────────────────
        velocity = self._estimate_velocity(delta, timestamp)

        # ── Force estimation via calibration curve ────────────────────
        force = max(0.0, self._slope * delta + self._intercept)

        # ── Regime classification ─────────────────────────────────────
        regime = self._classify_regime(delta)

        return GripperState(
            centroid=centroid,
            rest_pose=self._rest,
            deflection_px=delta,
            force_N=force,
            contact=(delta > DEFLECTION_CONTACT),
            regime=regime,
            velocity_px_s=velocity,
        )

    def has_rest_pose(self) -> bool:
        return self._rest is not None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _classify_regime(self, delta: float) -> str:
        if delta < DEFLECTION_CONTACT:
            return "free"
        elif delta < DEFLECTION_SOFT_STOP:
            return "contact"
        elif delta < DEFLECTION_HARD_STOP:
            return "soft_stop"
        else:
            return "hard_stop"

    def _estimate_velocity(
        self, delta: float, timestamp: Optional[float]
    ) -> float:
        """Exponential Moving Average of dδ/dt."""
        import time
        now = timestamp if timestamp is not None else time.perf_counter()

        if self._prev_time is None:
            self._prev_time = now
            self._prev_delta = delta
            return 0.0

        dt = now - self._prev_time
        if dt <= 0:
            return self._velocity_filtered

        raw_velocity = (delta - self._prev_delta) / dt
        self._velocity_filtered = (
            self._alpha * self._velocity_filtered
            + (1 - self._alpha) * raw_velocity
        )
        self._prev_delta = delta
        self._prev_time  = now
        return self._velocity_filtered

    def _save_rest_pose(self, pose: Tuple[float, float]) -> None:
        os.makedirs(os.path.dirname(_REST_POSE_FILE), exist_ok=True)
        with open(_REST_POSE_FILE, "w") as f:
            json.dump({"P0x": pose[0], "P0y": pose[1]}, f)

    def _try_load_rest_pose(self) -> None:
        if os.path.exists(_REST_POSE_FILE):
            with open(_REST_POSE_FILE) as f:
                d = json.load(f)
            self._rest = (d["P0x"], d["P0y"])
            print(f"[StateEstimator] Loaded rest pose from disk → {self._rest}")
