"""
========================================================================
  control/pid_controller.py
  Stage C — Control: PID Force Controller (Visual Servoing Loop)
========================================================================
  Implements a discrete-time PID controller that maps visual deflection
  error to a PWM command for the gripper's DC motor.

  The "set-point" is the TARGET deflection (default 0 = grip until
  contact, then hold).  In practice, for a grasping experiment you set
  a target deflection corresponding to a desired virtual force.

  Anti-windup: integral is clamped to avoid accumulation during saturation.
  Velocity feed-forward: uses dδ/dt from StateEstimator to anticipate
  rapid deformation events and cut speed proactively.
"""

import time
from dataclasses import dataclass
from typing import Tuple
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import (
    PID_KP, PID_KI, PID_KD,
    PID_SETPOINT, PID_OUTPUT_LIMITS,
    DEFLECTION_SOFT_STOP, DEFLECTION_HARD_STOP,
    PWM_MAX, PWM_MIN
)


@dataclass
class ControlOutput:
    """Result of one PID computation."""
    pwm: int            # PWM duty cycle [0–255] to send to motor driver
    error: float        # setpoint − deflection (px)
    p_term: float       # proportional contribution
    i_term: float       # integral contribution
    d_term: float       # derivative contribution
    saturated: bool     # True if output was clamped
    state: str          # "closing" | "holding" | "emergency_stop"


class PIDController:
    """
    Discrete-time PID controller for adaptive grip force regulation.

    The controller operates in DEFLECTION SPACE (pixels), not force space,
    to avoid model errors from the force calibration curve propagating into
    the control loop.  The calibrated force is used only for logging and
    the safety hard-stop check.

    Parameters
    ----------
    kp, ki, kd     : PID gains (tune empirically; see docs/tuning_guide.md)
    setpoint       : target deflection in pixels
    output_limits  : (min, max) PWM output clamp
    ff_velocity    : velocity feed-forward gain (px/s → PWM units)
    """

    def __init__(
        self,
        kp: float = PID_KP,
        ki: float = PID_KI,
        kd: float = PID_KD,
        setpoint: float = PID_SETPOINT,
        output_limits: Tuple[float, float] = PID_OUTPUT_LIMITS,
        ff_velocity: float = 0.5,   # feed-forward gain
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.output_min, self.output_max = output_limits
        self.ff_velocity = ff_velocity

        self._integral: float = 0.0
        self._prev_error: float = 0.0
        self._prev_time: float = time.perf_counter()

        # Anti-windup clamp for integral term
        self._integral_limit = (self.output_max - self.output_min) / self.ki \
            if self.ki != 0 else 1e6

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        deflection_px: float,
        velocity_px_s: float = 0.0,
        regime: str = "free",
    ) -> ControlOutput:
        """
        Compute one PID step.

        Parameters
        ----------
        deflection_px : current δ from StateEstimator
        velocity_px_s : dδ/dt (filtered) for feed-forward term
        regime        : gripper regime string from StateEstimator

        Returns
        -------
        ControlOutput
        """
        now = time.perf_counter()
        dt  = now - self._prev_time
        dt  = max(dt, 1e-6)           # guard against dt=0 at startup

        # ── Safety overrides (before PID) ────────────────────────────
        if regime == "hard_stop" or deflection_px >= DEFLECTION_HARD_STOP:
            self._reset_integral()
            return ControlOutput(
                pwm=PWM_MIN, error=0, p_term=0, i_term=0, d_term=0,
                saturated=False, state="emergency_stop"
            )

        # ── Error ─────────────────────────────────────────────────────
        # In closing mode: setpoint=0 means "close until contact"
        # Error is negative while deflection is increasing → drives PWM up
        error = self.setpoint - deflection_px

        # ── Proportional ──────────────────────────────────────────────
        p_term = self.kp * error

        # ── Integral (with anti-windup clamp) ─────────────────────────
        self._integral += error * dt
        self._integral = max(
            -self._integral_limit,
            min(self._integral_limit, self._integral)
        )
        i_term = self.ki * self._integral

        # ── Derivative (error rate) ───────────────────────────────────
        d_term = self.kd * (error - self._prev_error) / dt

        # ── Velocity feed-forward (proactive braking) ─────────────────
        # If δ is growing fast, reduce PWM proportionally before PID reacts
        ff_term = -self.ff_velocity * max(0.0, velocity_px_s)

        # ── Raw output ────────────────────────────────────────────────
        raw = p_term + i_term + d_term + ff_term

        # ── Soft-stop zone: cap PWM at 40% of max ────────────────────
        if regime == "soft_stop":
            effective_max = self.output_max * 0.4
            raw = min(raw, effective_max)

        # ── Clamp & determine saturation ─────────────────────────────
        pwm_float  = max(self.output_min, min(self.output_max, raw))
        saturated  = (pwm_float == self.output_min or
                      pwm_float == self.output_max)

        # ── State label ───────────────────────────────────────────────
        if regime in ("soft_stop", "contact") and abs(error) < 2.0:
            state = "holding"
        else:
            state = "closing"

        # ── Update history ────────────────────────────────────────────
        self._prev_error = error
        self._prev_time  = now

        return ControlOutput(
            pwm=int(round(pwm_float)),
            error=error,
            p_term=p_term,
            i_term=i_term,
            d_term=d_term,
            saturated=saturated,
            state=state,
        )

    def reset(self) -> None:
        """Reset controller state (call between grasp attempts)."""
        self._integral   = 0.0
        self._prev_error = 0.0
        self._prev_time  = time.perf_counter()

    def set_setpoint(self, sp: float) -> None:
        """Update target deflection at runtime."""
        self.setpoint = sp

    # ------------------------------------------------------------------

    def _reset_integral(self) -> None:
        self._integral = 0.0
