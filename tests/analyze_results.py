"""
========================================================================
  tests/analyze_results.py
  Post-Experiment Analysis — Generate Publication-Quality Plots
========================================================================
  Reads the CSV log produced by main.py and outputs the figures
  typically required for a Q1/Q2 robotics paper:

    Figure 1: Deflection & Estimated Force vs. Time
    Figure 2: PWM Command vs. Time (PID response)
    Figure 3: Latency Distribution (detection-to-command delay)
    Figure 4: PID Term Decomposition (P, I, D contributions)
    Figure 5: Phase Portrait — Force vs. Deflection rate

  Usage:
    python tests/analyze_results.py --log data/logs/run_20240101_120000.csv
    python tests/analyze_results.py          # uses most recent log file
"""

import argparse
import os
import glob
import sys
import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.ticker import MultipleLocator
except ImportError:
    print("[ERROR] matplotlib not installed: pip install matplotlib")
    sys.exit(1)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import (
    DEFLECTION_CONTACT, DEFLECTION_SOFT_STOP, DEFLECTION_HARD_STOP,
    CALIB_SLOPE, CALIB_INTERCEPT
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "figures")

# Publication style
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        10,
    "axes.titlesize":   11,
    "axes.labelsize":   10,
    "legend.fontsize":  9,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "figure.dpi":       150,
    "lines.linewidth":  1.4,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "axes.spines.top":  False,
    "axes.spines.right": False,
})


def load_log(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalise elapsed time to start at 0
    df["t"] = df["elapsed_s"] - df["elapsed_s"].iloc[0]
    return df


def find_latest_log() -> str:
    logs = glob.glob("data/logs/*.csv")
    if not logs:
        raise FileNotFoundError("No log files found in data/logs/")
    return max(logs, key=os.path.getmtime)


# ── Figure 1: Deflection & Force over time ────────────────────────────

def plot_deflection_force(df: pd.DataFrame, out_dir: str):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)

    ax1.plot(df["t"], df["deflection_px"], color="#1a73e8", label="δ (px)")
    ax1.axhline(DEFLECTION_CONTACT,   ls="--", color="green",
                lw=0.9, label=f"Contact ({DEFLECTION_CONTACT} px)")
    ax1.axhline(DEFLECTION_SOFT_STOP, ls="--", color="orange",
                lw=0.9, label=f"Soft-stop ({DEFLECTION_SOFT_STOP} px)")
    ax1.axhline(DEFLECTION_HARD_STOP, ls="--", color="red",
                lw=0.9, label=f"Hard-stop ({DEFLECTION_HARD_STOP} px)")
    ax1.set_ylabel("Deflection δ (px)")
    ax1.legend(loc="upper right", framealpha=0.7)

    ax2.plot(df["t"], df["force_N"], color="#e84c1a", label="F̂ (N)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Estimated Force (N)")
    ax2.legend(loc="upper right", framealpha=0.7)

    fig.suptitle("Finger Deflection and Estimated Force vs. Time",
                 fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "fig1_deflection_force.pdf")
    plt.savefig(path, bbox_inches="tight")
    plt.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close()


# ── Figure 2: PWM Command (PID response) ─────────────────────────────

def plot_pwm(df: pd.DataFrame, out_dir: str):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(df["t"], df["pwm"], color="#555", label="PWM command")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("PWM Duty Cycle [0–255]")
    ax.set_title("Motor PWM Command (PID Output) vs. Time", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, "fig2_pwm_command.pdf")
    plt.savefig(path, bbox_inches="tight")
    plt.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close()


# ── Figure 3: Latency Distribution ────────────────────────────────────

def plot_latency(df: pd.DataFrame, out_dir: str):
    lat = df["latency_ms"].dropna()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(lat, bins=40, color="#1a73e8", edgecolor="white", linewidth=0.4)
    ax.axvline(lat.mean(), color="red", ls="--", lw=1.2,
               label=f"Mean = {lat.mean():.2f} ms")
    ax.axvline(lat.quantile(0.95), color="orange", ls="--", lw=1.2,
               label=f"P95 = {lat.quantile(0.95):.2f} ms")
    ax.set_xlabel("Detection Latency (ms)")
    ax.set_ylabel("Frame Count")
    ax.set_title("Detection-to-Command Latency Distribution", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, "fig3_latency_distribution.pdf")
    plt.savefig(path, bbox_inches="tight")
    plt.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close()


# ── Figure 4: PID Term Decomposition ─────────────────────────────────

def plot_pid_terms(df: pd.DataFrame, out_dir: str):
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(df["t"], df["pid_p"], label="P term", color="#1a73e8")
    ax.plot(df["t"], df["pid_i"], label="I term", color="#e84c1a")
    ax.plot(df["t"], df["pid_d"], label="D term", color="#188038")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("PID Contribution (PWM units)")
    ax.set_title("PID Controller Term Decomposition", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, "fig4_pid_terms.pdf")
    plt.savefig(path, bbox_inches="tight")
    plt.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close()


# ── Figure 5: Phase Portrait (Force vs. dδ/dt) ────────────────────────

def plot_phase_portrait(df: pd.DataFrame, out_dir: str):
    # Estimate velocity from logged deflection (numerical diff)
    delta    = df["deflection_px"].values
    dt_arr   = np.diff(df["elapsed_s"].values)
    velocity = np.concatenate([[0], np.diff(delta) / np.where(dt_arr > 0, dt_arr, 1e-6)])

    fig, ax = plt.subplots(figsize=(5, 5))
    sc = ax.scatter(velocity, df["force_N"], c=df["t"], cmap="viridis",
                    s=5, alpha=0.6)
    plt.colorbar(sc, ax=ax, label="Time (s)")
    ax.axvline(0, color="black", lw=0.6)
    ax.set_xlabel("Deflection Rate dδ/dt (px/s)")
    ax.set_ylabel("Estimated Force F̂ (N)")
    ax.set_title("Phase Portrait: Force vs. Deflection Rate", fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "fig5_phase_portrait.pdf")
    plt.savefig(path, bbox_inches="tight")
    plt.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close()


# ── Summary statistics ────────────────────────────────────────────────

def print_summary(df: pd.DataFrame):
    print("\n=== EXPERIMENT SUMMARY ===")
    print(f"  Duration         : {df['t'].iloc[-1]:.2f} s")
    print(f"  Total frames     : {len(df)}")
    print(f"  Avg FPS          : {len(df) / df['t'].iloc[-1]:.1f}")
    print(f"  Max deflection   : {df['deflection_px'].max():.2f} px")
    print(f"  Max force (est.) : {df['force_N'].max():.4f} N")
    print(f"  Latency mean     : {df['latency_ms'].mean():.2f} ms")
    print(f"  Latency P95      : {df['latency_ms'].quantile(0.95):.2f} ms")
    print(f"  Marker loss rate : "
          f"{(1 - df['marker_found'].mean()) * 100:.1f}%")
    print("==========================\n")


# ── Entry point ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analyse visual gripper experiment log."
    )
    parser.add_argument("--log", type=str, default=None,
                        help="Path to CSV log file (default: latest)")
    args = parser.parse_args()

    log_path = args.log if args.log else find_latest_log()
    print(f"Loading: {log_path}")
    df = load_log(log_path)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Saving figures to: {OUTPUT_DIR}\n")

    print_summary(df)
    plot_deflection_force(df, OUTPUT_DIR)
    plot_pwm(df, OUTPUT_DIR)
    plot_latency(df, OUTPUT_DIR)
    plot_pid_terms(df, OUTPUT_DIR)
    plot_phase_portrait(df, OUTPUT_DIR)

    print("\nDone. All figures saved as PDF + PNG.")


if __name__ == "__main__":
    main()
