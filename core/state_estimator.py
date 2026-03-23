"""
core/state_estimator.py — Etapa B del Pipeline: Estimación de Estado (Propiocepción)

Responsabilidades:
  1. Mantener la posición de reposo P0 del marcador
  2. Calcular la deflexión euclidiana δ en píxeles
  3. Traducir δ a Fuerza F en Newtons mediante la curva de calibración
  4. Calcular la velocidad de deflexión dδ/dt (crítica para el control derivativo)

Referencia de la ecuación:
    δ = sqrt((Px - P0x)² + (Py - P0y)²)       [píxeles]
    F = K * δ + b                               [Newtons]
"""

import numpy as np
import time
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


class StateEstimator:
    """
    Calcula el estado mecánico del dedo flexible a partir de la posición del marcador.

    Atributos públicos (actualizados en cada llamada a update()):
        delta_px    : Deflexión actual en píxeles (float)
        force_n     : Fuerza estimada en Newtons (float)
        delta_dot   : Velocidad de deflexión en px/s (float, positivo = aumentando)
        rest_pos    : Posición de reposo (P0x, P0y)
        is_calibrated: True si P0 fue establecida
    """

    def __init__(self):
        self.rest_pos: tuple | None = None
        self.is_calibrated          = False

        # Coeficientes de calibración F-δ (cargados desde archivo o config)
        self._load_calibration()

        # Estado interno para derivada temporal
        self._prev_delta    = 0.0
        self._prev_time     = time.perf_counter()

        # Historial para suavizado (media móvil de δ)
        self._delta_history = []
        self._history_size  = 5   # frames — ajustar según FPS

        # Métricas acumuladas (para publicación)
        self.peak_delta_px  = 0.0
        self.peak_force_n   = 0.0

    # ── Calibración ──────────────────────────────────────────────────────────

    def _load_calibration(self):
        """
        Carga coeficientes K y b desde archivo JSON si existe,
        de lo contrario usa los valores de config.py.
        """
        calib_path = os.path.join("data", "calibration.json")
        if os.path.exists(calib_path):
            with open(calib_path, "r") as f:
                d = json.load(f)
            self.K = d.get("K", config.CALIB_K)
            self.b = d.get("b", config.CALIB_B)
            print(f"[StateEstimator] Calibración cargada desde {calib_path}: "
                  f"K={self.K:.5f} N/px, b={self.b:.5f} N")
        else:
            self.K = config.CALIB_K
            self.b = config.CALIB_B
            print(f"[StateEstimator] Usando calibración de config.py: "
                  f"K={self.K:.5f} N/px, b={self.b:.5f} N")

    def set_rest_position(self, centroid: tuple):
        """
        Establece P0 — la posición de reposo del marcador (sin contacto).

        Args:
            centroid: (x, y) en píxeles del marcador en posición libre.
        """
        self.rest_pos      = centroid
        self.is_calibrated = True
        self._prev_delta   = 0.0
        self._delta_history.clear()
        print(f"[StateEstimator] Posición de reposo establecida: P0 = {centroid}")

    def auto_calibrate_rest(self, centroids: list):
        """
        Calcula P0 promediando N frames con el gripper abierto.
        Llamar antes del experimento principal.

        Args:
            centroids: Lista de (x,y) válidos del marcador en reposo.
        """
        if len(centroids) < 5:
            raise ValueError("Se necesitan al menos 5 frames para auto-calibrar P0.")
        arr  = np.array(centroids, dtype=float)
        p0   = tuple(arr.mean(axis=0).astype(int))
        self.set_rest_position(p0)
        std  = arr.std(axis=0)
        print(f"[StateEstimator] Auto-calibración completada. "
              f"P0={p0}, std=[{std[0]:.2f}, {std[1]:.2f}] px")
        return p0

    # ── Estimación principal ─────────────────────────────────────────────────

    def update(self, centroid: tuple | None) -> dict:
        """
        Actualiza el estado con la posición actual del marcador.

        Args:
            centroid: (cx, cy) del marcador, o None si no fue detectado.

        Returns:
            state: Diccionario con todas las variables de estado:
                {
                  "delta_px"     : deflexión actual [px],
                  "delta_dot"    : velocidad de deflexión [px/s],
                  "force_n"      : fuerza estimada [N],
                  "detected"     : bool — marcador visible,
                  "timestamp"    : tiempo actual [s],
                  "at_warning"   : bool — supera umbral de advertencia,
                  "at_stop"      : bool — supera umbral de parada,
                }
        """
        now = time.perf_counter()
        dt  = now - self._prev_time
        dt  = max(dt, 1e-6)   # evitar división por cero

        if centroid is None or not self.is_calibrated:
            # Sin detección: retornar estado nulo pero mantener continuidad
            return self._null_state(now)

        # ── 1. Deflexión euclidiana ──────────────────────────────────────────
        p0  = self.rest_pos
        dx  = centroid[0] - p0[0]
        dy  = centroid[1] - p0[1]
        delta_raw = float(np.sqrt(dx**2 + dy**2))

        # ── 2. Suavizado por media móvil ─────────────────────────────────────
        self._delta_history.append(delta_raw)
        if len(self._delta_history) > self._history_size:
            self._delta_history.pop(0)
        delta_smooth = float(np.mean(self._delta_history))

        # ── 3. Velocidad de deflexión (dδ/dt) ────────────────────────────────
        delta_dot = (delta_smooth - self._prev_delta) / dt

        # ── 4. Traducción a Fuerza ────────────────────────────────────────────
        # F = K * δ + b  (regresión lineal del experimento de calibración)
        force_n = max(0.0, self.K * delta_smooth + self.b)

        # ── 5. Actualizar picos (para reporte de publicación) ────────────────
        self.peak_delta_px = max(self.peak_delta_px, delta_smooth)
        self.peak_force_n  = max(self.peak_force_n,  force_n)

        # ── 6. Guardar estado para próxima iteración ─────────────────────────
        self.delta_px  = delta_smooth
        self.force_n   = force_n
        self.delta_dot = delta_dot
        self._prev_delta = delta_smooth
        self._prev_time  = now

        return {
            "delta_px"  : delta_smooth,
            "delta_raw" : delta_raw,
            "delta_dot" : delta_dot,
            "force_n"   : force_n,
            "detected"  : True,
            "timestamp" : now,
            "centroid"  : centroid,
            "rest_pos"  : p0,
            "at_warning": delta_smooth >= config.DELTA_THRESHOLD_WARN_PX,
            "at_stop"   : delta_smooth >= config.DELTA_THRESHOLD_STOP_PX,
        }

    def _null_state(self, now: float) -> dict:
        """Estado vacío cuando no hay detección."""
        self._prev_time = now
        return {
            "delta_px"  : 0.0,
            "delta_raw" : 0.0,
            "delta_dot" : 0.0,
            "force_n"   : 0.0,
            "detected"  : False,
            "timestamp" : now,
            "centroid"  : None,
            "rest_pos"  : self.rest_pos,
            "at_warning": False,
            "at_stop"   : False,
        }

    def reset_peaks(self):
        """Reinicia picos acumulados (útil entre pruebas del experimento)."""
        self.peak_delta_px = 0.0
        self.peak_force_n  = 0.0

    def pixels_to_force(self, delta_px: float) -> float:
        """Función de transferencia pública para la curva de calibración."""
        return max(0.0, self.K * delta_px + self.b)

    def force_to_pixels(self, force_n: float) -> float:
        """Inversa: fuerza objetivo → deflexión esperada en píxeles."""
        return max(0.0, (force_n - self.b) / self.K)
