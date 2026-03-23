"""
========================================================================
  utils/data_logger.py
  Experiment Data Logger — CSV with timestamps
========================================================================
  Writes one row per control cycle with all variables needed for:
    - Latency analysis (detection → motor reaction time)
    - Force-deflection curves
    - PID performance plots
    - Peer-review reproducibility

  Output format (compatible with pandas / matplotlib / MATLAB):
    timestamp, frame_id, px, py, deflection_px, force_N,
    pwm, pid_error, pid_p, pid_i, pid_d, regime, state

  Usage
  -----
  with DataLogger("data/logs/run_001.csv") as log:
      log.write(timestamp=..., frame_id=..., ...)
"""

import csv
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import LOG_DIR, LOG_COLUMNS


class DataLogger:
    """
    Thread-safe CSV logger for one experiment run.

    Parameters
    ----------
    filename : output .csv path (auto-generated if None)
    """

    COLUMNS = [
        "timestamp_s",      # absolute time (perf_counter)
        "elapsed_s",        # time since logger opened
        "frame_id",         # monotonic frame counter
        "px",               # marker centroid x (px)
        "py",               # marker centroid y (px)
        "deflection_px",    # δ Euclidean distance (px)
        "force_N",          # estimated force (N)
        "pwm",              # motor PWM command [0–255]
        "pid_error",        # setpoint − deflection
        "pid_p",            # P term
        "pid_i",            # I term
        "pid_d",            # D term
        "regime",           # free/contact/soft_stop/hard_stop
        "ctrl_state",       # closing/holding/emergency_stop
        "marker_found",     # 1 = marker detected, 0 = lost
        "latency_ms",       # optional: detection-to-command latency
    ]

    def __init__(self, filename: Optional[str] = None):
        if filename is None:
            os.makedirs(LOG_DIR, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(LOG_DIR, f"run_{ts}.csv")

        self._filename = filename
        self._file     = None
        self._writer   = None
        self._t0       = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def open(self) -> "DataLogger":
        os.makedirs(os.path.dirname(self._filename), exist_ok=True)
        self._file   = open(self._filename, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.COLUMNS)
        self._writer.writeheader()
        self._t0 = time.perf_counter()
        print(f"[DataLogger] Logging → {self._filename}")
        return self

    def close(self) -> None:
        if self._file:
            self._file.flush()
            self._file.close()
            self._file = None
            print(f"[DataLogger] Closed → {self._filename}")

    def __enter__(self):
        return self.open()

    def __exit__(self, *_):
        self.close()

    # ------------------------------------------------------------------
    # Write one row
    # ------------------------------------------------------------------

    def write(
        self,
        frame_id: int,
        px: float,
        py: float,
        deflection_px: float,
        force_N: float,
        pwm: int,
        pid_error: float = 0.0,
        pid_p: float = 0.0,
        pid_i: float = 0.0,
        pid_d: float = 0.0,
        regime: str = "free",
        ctrl_state: str = "closing",
        marker_found: int = 1,
        latency_ms: float = 0.0,
    ) -> None:
        """Write one row to the CSV file."""
        if self._writer is None:
            raise RuntimeError("DataLogger not opened. Use as context manager.")

        now = time.perf_counter()
        row = {
            "timestamp_s":   now,
            "elapsed_s":     now - self._t0,
            "frame_id":      frame_id,
            "px":            round(px, 3),
            "py":            round(py, 3),
            "deflection_px": round(deflection_px, 3),
            "force_N":       round(force_N, 4),
            "pwm":           pwm,
            "pid_error":     round(pid_error, 3),
            "pid_p":         round(pid_p, 3),
            "pid_i":         round(pid_i, 3),
            "pid_d":         round(pid_d, 3),
            "regime":        regime,
            "ctrl_state":    ctrl_state,
            "marker_found":  marker_found,
            "latency_ms":    round(latency_ms, 3),
        }
        self._writer.writerow(row)

    @property
    def filename(self) -> str:
        return self._filename
