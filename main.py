"""
========================================================================
  main.py
  Visual Gripper — Main Experiment Loop
  "Adaptive Grasp Control via Proprioceptive Visual Servoing"
========================================================================
  Orchestrates the full perception → estimation → control pipeline
  at real-time frame rate.

  KEYBOARD CONTROLS (when DISPLAY_ENABLED=True):
    SPACE   — set rest pose (finger unloaded, gripper open)
    g       — start grasp sequence
    o       — open gripper (return to rest)
    r       — reset PID controller
    q / ESC — quit experiment

  PRE-FLIGHT CHECKLIST:
    1. Run  calibration/color_picker.py  → update MARKER_COLOR_* in settings.py
    2. Run  calibration/calibrate_force.py → update CALIB_SLOPE/INTERCEPT
    3. Check SERIAL_PORT in settings.py matches your Arduino port
    4. Gripper fully OPEN before pressing SPACE to set rest pose
"""

import cv2
import time
import sys
import os

# ── Local imports ──────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from config.settings   import DISPLAY_ENABLED, DISPLAY_SCALE, PWM_OPEN
from perception.color_tracker  import ColorMarkerTracker
from estimation.state_estimator import StateEstimator
from control.pid_controller    import PIDController
from control.motor_driver      import MotorDriver
from utils.camera              import Camera
from utils.data_logger         import DataLogger


# ── State machine for experiment flow ─────────────────────────────────
class GripperState:
    IDLE    = "IDLE"     # waiting for operator
    GRASPING = "GRASPING" # closing on object
    HOLDING = "HOLDING"  # grip maintained at target
    OPENING = "OPENING"  # returning to rest


def draw_hud(frame, gripper_state, cam_fps, state, ctrl_out, frame_id):
    """Draw heads-up display on the debug window."""
    h, w = frame.shape[:2]

    # Status bar background
    cv2.rectangle(frame, (0, h - 90), (w, h), (30, 30, 30), -1)

    delta_str = f"{state.deflection_px:.1f}" if state else "-.--"
    force_str = f"{state.force_N:.3f}"       if state else "-.---"
    regime_str = state.regime                 if state else "-"
    lines = [
        f"State: {gripper_state}   |  Regime: {regime_str}",
        f"d: {delta_str} px   F: {force_str} N   PWM: {ctrl_out.pwm if ctrl_out else 0}",
        f"FPS: {cam_fps:.1f}   Frame: {frame_id}   "
        f"[SPACE]=rest  [g]=grasp  [o]=open  [q]=quit",
    ]
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (8, h - 68 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1)

    # Force bar (top-left)
    if state:
        bar_max  = 100          # px width = 100% of max force
        force_pct = min(1.0, state.deflection_px / 45.0)
        color = (0, 255, 0) if force_pct < 0.5 else \
                (0, 165, 255) if force_pct < 0.8 else (0, 0, 255)
        cv2.rectangle(frame, (8, 8), (8 + int(bar_max * force_pct), 20),
                      color, -1)
        cv2.rectangle(frame, (8, 8), (8 + bar_max, 20), (180, 180, 180), 1)
        cv2.putText(frame, "DEFL", (115, 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)

    return frame


def run_experiment():
    # ── Component initialisation ───────────────────────────────────────
    tracker   = ColorMarkerTracker()
    estimator = StateEstimator()
    pid       = PIDController()
    motor     = MotorDriver()

    if not motor.open_connection():
        print("[WARN] Motor driver not available — running visual only.")

    frame_id    = 0
    gstate      = GripperState.IDLE
    last_state  = None
    last_ctrl   = None
    t_contact   = None          # timestamp of first contact detection

    with Camera() as cam, DataLogger() as log:
        print("\n=== EXPERIMENT RUNNING ===")
        print("Press SPACE with gripper OPEN to set rest pose.")
        print("Then press 'g' to start grasping.\n")

        while True:
            # ── 1. Acquire frame ──────────────────────────────────────
            t_frame_start = time.perf_counter()
            frame, ts = cam.read()
            if frame is None:
                print("[ERROR] Camera read failed.")
                break

            # ── 2. Perception — detect marker ─────────────────────────
            t_detect_start = time.perf_counter()
            obs = tracker.detect(frame)
            t_detect_end   = time.perf_counter()

            marker_found = obs is not None

            # ── 3. Estimation — gripper state ─────────────────────────
            state = None
            if marker_found and estimator.has_rest_pose():
                state = estimator.update(obs.centroid_global, ts)

                # Record time of first contact event
                if state.contact and t_contact is None:
                    t_contact = ts

            # ── 4. Control decision ───────────────────────────────────
            ctrl_out = None

            if gstate == GripperState.GRASPING and state is not None:
                ctrl_out = pid.compute(
                    deflection_px=state.deflection_px,
                    velocity_px_s=state.velocity_px_s,
                    regime=state.regime,
                )
                motor.set_pwm(ctrl_out.pwm)

                # Transition to HOLDING once stable contact established
                if ctrl_out.state == "holding":
                    gstate = GripperState.HOLDING

                # Emergency stop
                if ctrl_out.state == "emergency_stop":
                    motor.brake()
                    print("[SAFETY] Emergency stop — max deflection reached.")
                    gstate = GripperState.IDLE

            elif gstate == GripperState.HOLDING and state is not None:
                # Maintenance PID — keep grip stable
                ctrl_out = pid.compute(
                    deflection_px=state.deflection_px,
                    velocity_px_s=state.velocity_px_s,
                    regime=state.regime,
                )
                motor.set_pwm(ctrl_out.pwm)

            elif gstate == GripperState.OPENING:
                motor.set_pwm(PWM_OPEN, direction="open")
                # (In a real setup you'd detect the finger returning to P₀)
                time.sleep(0.5)
                motor.stop()
                pid.reset()
                t_contact = None
                gstate = GripperState.IDLE

            # ── 5. Data logging ───────────────────────────────────────
            if state is not None:
                latency = (t_detect_end - t_detect_start) * 1000  # ms
                log.write(
                    frame_id     = frame_id,
                    px           = state.centroid[0],
                    py           = state.centroid[1],
                    deflection_px= state.deflection_px,
                    force_N      = state.force_N,
                    pwm          = ctrl_out.pwm if ctrl_out else 0,
                    pid_error    = ctrl_out.error if ctrl_out else 0,
                    pid_p        = ctrl_out.p_term if ctrl_out else 0,
                    pid_i        = ctrl_out.i_term if ctrl_out else 0,
                    pid_d        = ctrl_out.d_term if ctrl_out else 0,
                    regime       = state.regime,
                    ctrl_state   = ctrl_out.state if ctrl_out else "idle",
                    marker_found = int(marker_found),
                    latency_ms   = latency,
                )
            frame_id += 1
            last_state = state
            last_ctrl  = ctrl_out

            # ── 6. Display ────────────────────────────────────────────
            if DISPLAY_ENABLED:
                annotated = tracker.draw_overlay(
                    frame, obs,
                    deflection_px = state.deflection_px if state else 0,
                    force_N       = state.force_N if state else 0,
                )
                annotated = draw_hud(
                    annotated, gstate, cam.fps_actual,
                    state, ctrl_out, frame_id
                )

                if DISPLAY_SCALE != 1.0:
                    dh = int(annotated.shape[0] * DISPLAY_SCALE)
                    dw = int(annotated.shape[1] * DISPLAY_SCALE)
                    annotated = cv2.resize(annotated, (dw, dh))

                cv2.imshow("Visual Gripper", annotated)

            # ── 7. Keyboard input ─────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF if DISPLAY_ENABLED else 0xFF

            if key in (ord('q'), 27):         # q / ESC
                break

            elif key == ord(' '):              # SPACE — set rest pose
                if marker_found:
                    estimator.set_rest_pose(obs.centroid_global)
                    print("[OK] Rest pose set.")
                else:
                    print("[WARN] Marker not detected — cannot set rest pose.")

            elif key == ord('g'):              # g — start grasp
                if not estimator.has_rest_pose():
                    print("[WARN] Set rest pose first (SPACE).")
                else:
                    gstate = GripperState.GRASPING
                    pid.reset()
                    print("[CMD] Grasping...")

            elif key == ord('o'):              # o — open
                gstate = GripperState.OPENING
                print("[CMD] Opening...")

            elif key == ord('r'):              # r — reset PID
                pid.reset()
                print("[CMD] PID reset.")

    # ── Cleanup ────────────────────────────────────────────────────────
    motor.stop()
    motor.close_connection()
    cv2.destroyAllWindows()
    print("\n=== EXPERIMENT ENDED ===")
    print(f"Data saved to: {log.filename}")


if __name__ == "__main__":
    run_experiment()
