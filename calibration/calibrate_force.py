"""
========================================================================
  calibration/calibrate_force.py
  Force–Deflection Calibration  (Run BEFORE the main experiment)
========================================================================
  PURPOSE
  -------
  Establishes the transfer function:  F(N) = k_s · δ(px) + b
  by recording pairs (δ_measured, F_reference) while pressing the gripper
  finger against a reference load cell or hanging known weights.

  PROCEDURE
  ---------
  1. Connect a calibrated load cell (or hang known weights) against the
     gripper finger tip.
  2. Run this script.  For each calibration step:
       - The motor closes until the camera detects deflection δ.
       - You enter the measured reference force F (in Newtons) when prompted.
  3. After N_STEPS samples, the script fits a linear regression and saves
     the coefficients to  data/calibration/force_calibration.json
  4. Copy the slope/intercept values into config/settings.py

  OUTPUT FILES
  ------------
  data/calibration/force_calibration.json  — fitted coefficients
  data/calibration/calib_data.csv          — raw (delta, force) pairs
  data/calibration/calib_plot.png          — scatter + regression line
"""

import cv2
import numpy as np
import json
import csv
import time
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from perception.color_tracker import ColorMarkerTracker
from estimation.state_estimator import StateEstimator
from utils.camera import Camera

# ── Calibration settings ─────────────────────────────────────────────
N_STEPS            = 10     # number of (force, deflection) data points
SAMPLE_FRAMES      = 30     # frames averaged per step to reduce noise
OUTPUT_DIR         = os.path.join(os.path.dirname(__file__),
                                   "..", "data", "calibration")

# ── Matplotlib (optional — skip if not installed) ────────────────────
try:
    import matplotlib.pyplot as plt
    _PLOT = True
except ImportError:
    _PLOT = False
    print("[Calib] matplotlib not found — skipping plot generation.")


def measure_deflection_stable(
    cam: Camera,
    tracker: ColorMarkerTracker,
    estimator: StateEstimator,
    n_frames: int = SAMPLE_FRAMES,
) -> float:
    """Average deflection over n_frames to reduce measurement noise."""
    deltas = []
    for _ in range(n_frames):
        frame, ts = cam.read()
        if frame is None:
            continue
        obs = tracker.detect(frame)
        if obs is None:
            continue
        state = estimator.update(obs.centroid_global, ts)
        deltas.append(state.deflection_px)
        time.sleep(0.02)

    if not deltas:
        return 0.0
    return float(np.mean(deltas))


def run_calibration():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tracker   = ColorMarkerTracker()
    estimator = StateEstimator()

    data_points = []   # list of (delta_px, force_N)

    with Camera() as cam:
        # ── Step 0: Set rest pose ─────────────────────────────────────
        print("\n=== CALIBRATION: Force–Deflection Characterisation ===")
        print("Step 0: Position the gripper fully OPEN with finger UNLOADED.")
        input("Press ENTER when ready...")

        # Collect a stable rest reading
        print("  Measuring rest pose (30 frames)...")
        rest_readings = []
        for _ in range(30):
            frame, ts = cam.read()
            if frame is None:
                continue
            obs = tracker.detect(frame)
            if obs:
                rest_readings.append(obs.centroid_global)
        if not rest_readings:
            print("[ERROR] Cannot detect marker at rest. Check HSV bounds.")
            return

        rest_pose = tuple(np.mean(rest_readings, axis=0))
        estimator.set_rest_pose(rest_pose)
        print(f"  Rest pose set → {rest_pose}")

        # ── Steps 1..N: Collect (δ, F) pairs ─────────────────────────
        for step in range(1, N_STEPS + 1):
            print(f"\nStep {step}/{N_STEPS}:")
            print("  Apply a known force to the finger tip.")
            print("  (Use a load cell, or hang a known weight over the tip.)")
            f_ref_str = input("  Enter reference force in Newtons [e.g. 0.5]: ")
            try:
                f_ref = float(f_ref_str)
            except ValueError:
                print("  Invalid input — skipping step.")
                continue

            print(f"  Measuring deflection while F = {f_ref} N is applied...")
            delta = measure_deflection_stable(cam, tracker, estimator)
            print(f"  → δ = {delta:.2f} px  |  F = {f_ref} N")
            data_points.append((delta, f_ref))

    if len(data_points) < 2:
        print("[ERROR] Need at least 2 data points for regression.")
        return

    # ── Linear regression: F = slope * δ + intercept ─────────────────
    deltas = np.array([d[0] for d in data_points])
    forces = np.array([d[1] for d in data_points])

    coeffs = np.polyfit(deltas, forces, 1)
    slope, intercept = float(coeffs[0]), float(coeffs[1])

    # R² coefficient of determination
    forces_pred = slope * deltas + intercept
    ss_res = np.sum((forces - forces_pred) ** 2)
    ss_tot = np.sum((forces - forces.mean()) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    print(f"\n=== CALIBRATION RESULT ===")
    print(f"  F = {slope:.4f} · δ + {intercept:.4f}")
    print(f"  R² = {r2:.4f}")
    print(f"  → Copy these values to config/settings.py:")
    print(f"      CALIB_SLOPE     = {slope:.4f}")
    print(f"      CALIB_INTERCEPT = {intercept:.4f}")

    # ── Save JSON ─────────────────────────────────────────────────────
    calib = {
        "slope": slope, "intercept": intercept, "r2": r2,
        "n_points": len(data_points),
        "data": [{"delta_px": d, "force_N": f} for d, f in data_points],
    }
    json_path = os.path.join(OUTPUT_DIR, "force_calibration.json")
    with open(json_path, "w") as jf:
        json.dump(calib, jf, indent=2)
    print(f"\n  Saved → {json_path}")

    # ── Save CSV ──────────────────────────────────────────────────────
    csv_path = os.path.join(OUTPUT_DIR, "calib_data.csv")
    with open(csv_path, "w", newline="") as cf:
        w = csv.writer(cf)
        w.writerow(["delta_px", "force_N"])
        w.writerows(data_points)
    print(f"  Saved → {csv_path}")

    # ── Plot ──────────────────────────────────────────────────────────
    if _PLOT:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(deltas, forces, s=60, zorder=3, label="Measurements")
        x_fit = np.linspace(0, deltas.max() * 1.1, 200)
        ax.plot(x_fit, slope * x_fit + intercept, "r-",
                label=f"F = {slope:.4f}·δ + {intercept:.4f}\n(R²={r2:.4f})")
        ax.set_xlabel("Deflection δ (px)")
        ax.set_ylabel("Force F (N)")
        ax.set_title("Force–Deflection Calibration Curve")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = os.path.join(OUTPUT_DIR, "calib_plot.png")
        plt.savefig(plot_path, dpi=150)
        print(f"  Plot   → {plot_path}")
        plt.show()


if __name__ == "__main__":
    run_calibration()
