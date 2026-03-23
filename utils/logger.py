"""
utils/logger.py — Logger CSV para datos experimentales

Registra todas las variables del experimento en tiempo real:
  timestamp, delta_px, force_n, pwm, action, error_pid, fps

Los CSVs generados son la fuente de datos para las figuras de la publicación.
Compatible con pandas, MATLAB, R, y Excel.
"""

import csv
import os
import time
from datetime import datetime


class ExperimentLogger:
    """
    Logger de alta frecuencia para datos del experimento.

    Escribe un CSV por sesión con todas las variables de estado y control.
    Soporta múltiples "trials" dentro del mismo experimento.

    Args:
        experiment_name: Prefijo del archivo (ej. "test_speed", "test_versatility")
        log_dir        : Directorio donde se guardan los CSVs
        interval_ms    : Mínimo intervalo entre registros en ms (0 = sin límite)
    """

    COLUMNS = [
        "timestamp_s",
        "trial",
        "object_name",
        "delta_px",
        "delta_raw_px",
        "delta_dot_px_s",
        "force_n",
        "pwm_percent",
        "pid_error_n",
        "pid_p",
        "pid_i",
        "pid_d",
        "action",
        "marker_detected",
        "fps",
        "notes",
    ]

    def __init__(self,
                 experiment_name: str = "experiment",
                 log_dir: str = "data",
                 interval_ms: float = 10.0):
        self.log_dir      = log_dir
        self.interval_ms  = interval_ms
        self._trial       = 1
        self._object_name = "unknown"
        self._start_time  = time.perf_counter()
        self._last_log_t  = 0.0
        self._row_count   = 0

        os.makedirs(log_dir, exist_ok=True)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename      = f"{experiment_name}_{timestamp_str}.csv"
        self.filepath = os.path.join(log_dir, filename)

        self._file   = open(self.filepath, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.COLUMNS)
        self._writer.writeheader()

        print(f"[Logger] Registrando en: {self.filepath}")

    def start_trial(self, trial_number: int, object_name: str = ""):
        """Marca el inicio de un nuevo trial (objeto diferente, repetición, etc.)."""
        self._trial       = trial_number
        self._object_name = object_name
        self._start_time  = time.perf_counter()
        print(f"[Logger] Trial {trial_number} iniciado — objeto: '{object_name}'")

    def log(self, state: dict, control: dict, fps: float = 0.0, notes: str = ""):
        """
        Registra un ciclo del experimento.

        Args:
            state  : Dict de StateEstimator.update()
            control: Dict de PIDController.compute()
            fps    : FPS del pipeline en este ciclo
            notes  : Anotación libre (ej. "contacto inicial", "objeto soltado")
        """
        now = time.perf_counter()
        if (now - self._last_log_t) * 1000 < self.interval_ms:
            return   # Limitar frecuencia de escritura

        row = {
            "timestamp_s"   : round(now - self._start_time, 5),
            "trial"         : self._trial,
            "object_name"   : self._object_name,
            "delta_px"      : round(state.get("delta_px",    0.0), 3),
            "delta_raw_px"  : round(state.get("delta_raw",   0.0), 3),
            "delta_dot_px_s": round(state.get("delta_dot",   0.0), 3),
            "force_n"       : round(state.get("force_n",     0.0), 5),
            "pwm_percent"   : round(control.get("pwm",        0.0), 2),
            "pid_error_n"   : round(control.get("error_n",    0.0), 5),
            "pid_p"         : round(control.get("p_term",     0.0), 5),
            "pid_i"         : round(control.get("i_term",     0.0), 5),
            "pid_d"         : round(control.get("d_term",     0.0), 5),
            "action"        : control.get("action", ""),
            "marker_detected": int(state.get("detected", False)),
            "fps"           : round(fps, 2),
            "notes"         : notes,
        }
        self._writer.writerow(row)
        self._last_log_t = now
        self._row_count += 1

    def flush(self):
        """Fuerza escritura al disco (usar al terminar cada trial)."""
        self._file.flush()

    def close(self):
        """Cierra el archivo de log."""
        self._file.flush()
        self._file.close()
        print(f"[Logger] Cerrado. {self._row_count} filas registradas → {self.filepath}")

    def annotate(self, note: str):
        """Agrega una nota en el próximo registro (evento importante)."""
        print(f"[Logger] Anotación: {note}")
        # Se pasa como 'notes' en el próximo log()
        self._pending_note = note

    @property
    def filepath_str(self):
        return self.filepath
