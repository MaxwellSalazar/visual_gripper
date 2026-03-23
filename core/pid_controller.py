"""
core/pid_controller.py — Etapa C del Pipeline: Controlador PID de Fuerza Virtual

Responsabilidades:
  1. Implementar un PID discreto robusto con anti-windup
  2. Traducir el error de fuerza (F_objetivo - F_medida) → señal PWM [0-100%]
  3. Implementar las reglas de seguridad por deflexión crítica
  4. Registrar el historial del controlador para análisis de la publicación

El PID opera sobre la FUERZA VIRTUAL (derivada de δ), no sobre el PWM directamente.
Esto garantiza que el control sea independiente del hardware del motor.

Ecuación discreta:
    u[k] = Kp*e[k] + Ki*Σe[k]*dt + Kd*(e[k]-e[k-1])/dt
"""

import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


class PIDController:
    """
    Controlador PID discreto con:
      - Anti-windup por saturación del término integral
      - Filtro derivativo (suaviza el término D para evitar kick)
      - Reglas de seguridad de emergencia por hardware

    Args:
        kp, ki, kd   : Ganancias del PID (por defecto desde config.py)
        setpoint     : Fuerza objetivo en Newtons
        output_limits: (min, max) del PWM de salida [%]
    """

    def __init__(self,
                 kp=None, ki=None, kd=None,
                 setpoint=None,
                 output_limits=(config.PWM_MIN, config.PWM_MAX)):

        self.kp = kp if kp is not None else config.PID_KP
        self.ki = ki if ki is not None else config.PID_KI
        self.kd = kd if kd is not None else config.PID_KD

        self.setpoint     = setpoint if setpoint is not None else config.FORCE_SETPOINT_N
        self.output_min   = output_limits[0]
        self.output_max   = output_limits[1]

        # Estado interno
        self._integral    = 0.0
        self._prev_error  = 0.0
        self._prev_time   = time.perf_counter()
        self._first_call  = True

        # Filtro derivativo: alpha ∈ [0,1] — 0=muy suave, 1=sin filtro
        self._alpha_d     = 0.3
        self._filtered_d  = 0.0

        # Anti-windup: límite del término integral para evitar saturación
        self._integral_max = (self.output_max - self.output_min) / self.ki \
                             if self.ki > 0 else float('inf')

        # Historial para análisis
        self.history = []   # Lista de dicts por ciclo

    # ── Control principal ────────────────────────────────────────────────────

    def compute(self, state: dict) -> dict:
        """
        Calcula la señal de control (PWM) a partir del estado actual.

        Args:
            state: Diccionario del StateEstimator con keys:
                   delta_px, delta_dot, force_n, at_warning, at_stop, detected

        Returns:
            control: {
                "pwm"          : Señal PWM [0-100%],
                "action"       : "run" | "slow" | "stop" | "no_detection",
                "error_n"      : Error de fuerza [N],
                "p_term"       : Término proporcional,
                "i_term"       : Término integral,
                "d_term"       : Término derivativo,
                "setpoint_n"   : Fuerza objetivo,
            }
        """
        now = time.perf_counter()
        dt  = now - self._prev_time
        dt  = max(dt, 1e-4)   # mínimo 0.1 ms

        # ── Reglas de seguridad (mayor prioridad que el PID) ─────────────────
        if not state["detected"]:
            # Sin marcador visible: política conservadora (reducir velocidad)
            self._reset_integrator()
            result = self._make_result(
                pwm=config.PWM_INITIAL * 0.5,
                action="no_detection",
                error=0.0, p=0.0, i=0.0, d=0.0
            )
            self._log(result, state, now)
            return result

        if state["at_stop"]:
            # Umbral crítico superado: parada inmediata
            self._reset_integrator()
            result = self._make_result(
                pwm=0.0, action="stop",
                error=0.0, p=0.0, i=0.0, d=0.0
            )
            self._log(result, state, now)
            self._prev_time = now
            return result

        # ── Cálculo PID ───────────────────────────────────────────────────────
        force_measured = state["force_n"]
        error          = self.setpoint - force_measured

        # Término Proporcional
        p_term = self.kp * error

        # Término Integral con anti-windup
        self._integral += error * dt
        self._integral  = max(-self._integral_max,
                               min(self._integral_max, self._integral))
        i_term = self.ki * self._integral

        # Término Derivativo sobre la MEDICIÓN (no el error) para evitar
        # derivative kick al cambiar el setpoint
        if self._first_call:
            raw_d          = 0.0
            self._first_call = False
        else:
            raw_d = -self.kd * (force_measured - self._prev_error) / dt

        # Filtro de primer orden sobre D
        self._filtered_d = (self._alpha_d * raw_d
                            + (1 - self._alpha_d) * self._filtered_d)
        d_term = self._filtered_d

        # ── Suma PID ──────────────────────────────────────────────────────────
        # Nota: el motor CIERRA el gripper. El PWM base es el de movimiento.
        # Si la fuerza es menor al setpoint → aumentamos PWM (más cierre).
        # Si la fuerza supera el setpoint → reducimos PWM.
        raw_pwm = config.PWM_INITIAL + p_term + i_term + d_term

        # Saturación de salida
        pwm = max(self.output_min, min(self.output_max, raw_pwm))

        # Reducción adicional si se acerca al umbral de advertencia
        if state["at_warning"]:
            pwm = min(pwm, config.PWM_INITIAL * 0.4)
            action = "slow"
        else:
            action = "run"

        # Guardar estado para siguiente iteración
        self._prev_error = force_measured
        self._prev_time  = now

        result = self._make_result(
            pwm=pwm, action=action,
            error=error, p=p_term, i=i_term, d=d_term
        )
        self._log(result, state, now)
        return result

    # ── Utilidades ────────────────────────────────────────────────────────────

    def _make_result(self, pwm, action, error, p, i, d) -> dict:
        return {
            "pwm"       : pwm,
            "action"    : action,
            "error_n"   : error,
            "p_term"    : p,
            "i_term"    : i,
            "d_term"    : d,
            "setpoint_n": self.setpoint,
        }

    def _reset_integrator(self):
        """Reinicia el integrador para evitar windup en paradas largas."""
        self._integral   = 0.0
        self._filtered_d = 0.0

    def _log(self, result: dict, state: dict, timestamp: float):
        """Guarda historial para análisis post-experimento."""
        self.history.append({
            "t"         : timestamp,
            "delta_px"  : state.get("delta_px", 0),
            "force_n"   : state.get("force_n", 0),
            "error_n"   : result["error_n"],
            "pwm"       : result["pwm"],
            "action"    : result["action"],
            "p"         : result["p_term"],
            "i"         : result["i_term"],
            "d"         : result["d_term"],
        })

    def tune(self, kp=None, ki=None, kd=None):
        """Permite cambiar ganancias en caliente (útil durante experimentos)."""
        if kp is not None: self.kp = kp
        if ki is not None:
            self.ki = ki
            self._integral_max = (self.output_max - self.output_min) / ki \
                                 if ki > 0 else float('inf')
        if kd is not None: self.kd = kd
        print(f"[PID] Ganancias actualizadas → Kp={self.kp}, Ki={self.ki}, Kd={self.kd}")

    def reset(self):
        """Reinicia el controlador completo (entre pruebas)."""
        self._integral    = 0.0
        self._prev_error  = 0.0
        self._filtered_d  = 0.0
        self._first_call  = True
        self._prev_time   = time.perf_counter()
        self.history.clear()
        print("[PID] Controlador reiniciado.")

    def get_summary(self) -> dict:
        """Resumen estadístico del experimento para la publicación."""
        if not self.history:
            return {}
        pwms   = [h["pwm"]     for h in self.history]
        forces = [h["force_n"] for h in self.history]
        errors = [h["error_n"] for h in self.history]
        return {
            "n_cycles"    : len(self.history),
            "pwm_mean"    : sum(pwms) / len(pwms),
            "pwm_min"     : min(pwms),
            "pwm_max"     : max(pwms),
            "force_mean"  : sum(forces) / len(forces),
            "force_max"   : max(forces),
            "error_rms"   : (sum(e**2 for e in errors) / len(errors)) ** 0.5,
        }
