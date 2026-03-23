"""
utils/visualizer.py — Overlay visual en tiempo real

Genera el HUD (Heads-Up Display) sobre el frame de la cámara con:
  - Deflexión δ y fuerza F en tiempo real
  - Barra de progreso del PWM
  - Gráfica de fuerza scrolling (últimos N segundos)
  - Estado del PID y alertas de seguridad

Las capturas de pantalla de esta ventana son utilizables directamente
en figuras de la publicación (se recomienda grabar en MJPG a 60fps).
"""

import cv2
import numpy as np
import collections
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


class RealtimeVisualizer:
    """
    Renderiza el overlay de diagnóstico sobre el frame del experimento.

    Args:
        history_seconds: Segundos de historial en la gráfica scrolling
        fps_display    : Mostrar FPS en el overlay
    """

    def __init__(self, history_seconds: float = 5.0):
        # Colas de historial para mini-gráfica
        max_pts = int(history_seconds * config.CAMERA_FPS)
        self._force_hist = collections.deque(maxlen=max_pts)
        self._delta_hist = collections.deque(maxlen=max_pts)
        self._pwm_hist   = collections.deque(maxlen=max_pts)
        self._t_start    = time.perf_counter()

        # Paleta de colores
        self.C_OK      = (80,  200, 80)    # Verde — normal
        self.C_WARN    = (30,  180, 255)   # Naranja — advertencia
        self.C_STOP    = (40,  40,  220)   # Rojo — parada
        self.C_TEXT    = (230, 230, 230)
        self.C_BG      = (20,  20,  20)

    def render(self, frame: np.ndarray, state: dict, control: dict,
               fps: float = 0.0) -> np.ndarray:
        """
        Renderiza el overlay completo sobre el frame.

        Args:
            frame  : Frame BGR de la cámara (ya anotado por perception.py)
            state  : Dict de StateEstimator
            control: Dict de PIDController
            fps    : FPS del pipeline

        Returns:
            output: Frame con overlay completo
        """
        output = frame.copy()
        H, W   = output.shape[:2]

        # Actualizar historiales
        self._force_hist.append(state.get("force_n",  0.0))
        self._delta_hist.append(state.get("delta_px", 0.0))
        self._pwm_hist.append(control.get("pwm",      0.0))

        # Determinar color de estado
        if state.get("at_stop"):
            c_status = self.C_STOP
            txt_status = "!! PARADA DE EMERGENCIA !!"
        elif state.get("at_warning"):
            c_status = self.C_WARN
            txt_status = "ADVERTENCIA — Reduciendo"
        elif not state.get("detected"):
            c_status = (100, 100, 100)
            txt_status = "Sin detección de marcador"
        else:
            c_status = self.C_OK
            txt_status = f"Acción: {control.get('action', '—').upper()}"

        # ── Panel superior ────────────────────────────────────────────────────
        self._draw_top_panel(output, state, control, fps, c_status, txt_status)

        # ── Barra lateral derecha: métricas ──────────────────────────────────
        self._draw_side_metrics(output, state, control, W, H)

        # ── Mini-gráfica de fuerza (esquina inferior izquierda) ───────────────
        self._draw_scrolling_plot(output, H, W)

        return output

    # ── Panels internos ───────────────────────────────────────────────────────

    def _draw_top_panel(self, img, state, control, fps,
                        c_status, txt_status):
        """Barra superior con métricas principales."""
        # Fondo semitransparente
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (img.shape[1], 45), self.C_BG, -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

        delta = state.get("delta_px", 0.0)
        force = state.get("force_n",  0.0)
        pwm   = control.get("pwm",    0.0)

        line1 = (f"δ={delta:5.1f}px  |  F={force:.3f}N  |  "
                 f"PWM={pwm:5.1f}%  |  FPS={fps:.1f}  |  {txt_status}")

        cv2.putText(img, line1, (8, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, c_status, 1,
                    cv2.LINE_AA)

    def _draw_side_metrics(self, img, state, control, W, H):
        """Panel derecho con barras de progreso verticales."""
        pw  = 28     # ancho de cada barra
        ph  = 120    # alto de la barra
        x0  = W - 130
        y0  = 55

        metrics = [
            ("δ",   state.get("delta_px", 0) / config.DELTA_THRESHOLD_STOP_PX,
             self.C_WARN),
            ("F",   min(state.get("force_n", 0) /
                        max(config.FORCE_SETPOINT_N * 2, 0.01), 1.0),
             self.C_OK),
            ("PWM", control.get("pwm", 0) / 100.0, (200, 130, 50)),
        ]

        for i, (label, ratio, color) in enumerate(metrics):
            xi = x0 + i * (pw + 10)
            # Marco
            cv2.rectangle(img, (xi, y0), (xi + pw, y0 + ph),
                          (80, 80, 80), 1)
            # Relleno
            fill_h = int(ph * min(max(ratio, 0), 1))
            cv2.rectangle(img,
                          (xi + 1, y0 + ph - fill_h),
                          (xi + pw - 1, y0 + ph),
                          color, -1)
            # Etiqueta
            cv2.putText(img, label,
                        (xi + 2, y0 + ph + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        self.C_TEXT, 1)
            cv2.putText(img, f"{int(ratio*100)}%",
                        (xi - 2, y0 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        self.C_TEXT, 1)

    def _draw_scrolling_plot(self, img, H, W):
        """Mini-gráfica de fuerza vs tiempo en la esquina inferior."""
        if len(self._force_hist) < 2:
            return

        gw, gh  = 200, 80   # ancho y alto de la gráfica
        gx, gy  = 8, H - gh - 8

        # Fondo
        overlay = img.copy()
        cv2.rectangle(overlay, (gx, gy), (gx + gw, gy + gh),
                      self.C_BG, -1)
        cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
        cv2.rectangle(img, (gx, gy), (gx + gw, gy + gh), (80, 80, 80), 1)

        # Línea de setpoint
        sp_ratio = config.FORCE_SETPOINT_N / max(
            max(self._force_hist) * 1.2, 0.01)
        sp_y = int(gy + gh - sp_ratio * gh)
        cv2.line(img, (gx, sp_y), (gx + gw, sp_y), (80, 180, 80), 1)

        # Curva de fuerza
        forces = list(self._force_hist)
        n      = len(forces)
        f_max  = max(max(forces) * 1.2, config.FORCE_SETPOINT_N * 2, 0.01)
        pts    = []
        for i, f in enumerate(forces):
            px = gx + int(i / n * gw)
            py = gy + gh - int((f / f_max) * gh)
            pts.append((px, py))

        for i in range(1, len(pts)):
            cv2.line(img, pts[i-1], pts[i], self.C_OK, 1, cv2.LINE_AA)

        # Etiqueta
        cv2.putText(img, "F [N]", (gx + 2, gy + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.C_TEXT, 1)
        cv2.putText(img,
                    f"max:{max(forces):.3f}N",
                    (gx + 2, gy + gh - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, self.C_TEXT, 1)

    def reset(self):
        """Limpia el historial (al cambiar de trial)."""
        self._force_hist.clear()
        self._delta_hist.clear()
        self._pwm_hist.clear()
