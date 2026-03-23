"""
config.py — Parámetros centralizados del sistema Visual Soft Gripper
Modifica aquí sin tocar el código de los módulos.
"""

# ─── CÁMARA ───────────────────────────────────────────────────────────────────
CAMERA_INDEX     = 0          # 0 = cámara por defecto / PiCam
CAMERA_WIDTH     = 640
CAMERA_HEIGHT    = 480
CAMERA_FPS       = 60         # Mínimo recomendado para baja latencia

# Región de Interés (ROI) — [x, y, ancho, alto] en píxeles
# Ajustar una vez que el gripper esté montado
ROI = [160, 100, 320, 280]   # [x_start, y_start, width, height]

# ─── MARCADOR DE COLOR (HSV) ──────────────────────────────────────────────────
# Definir con calibrate_marker.py primero, luego copiar aquí
MARKER_HSV_LOWER = [35, 100, 80]   # Verde por defecto
MARKER_HSV_UPPER = [85, 255, 255]

MARKER_MIN_AREA  = 200   # Área mínima del blob en píxeles² (filtra ruido)

# ─── POSICIÓN DE REPOSO ───────────────────────────────────────────────────────
# Se calcula automáticamente al inicio, pero puede forzarse aquí
REST_POSITION_PX = None   # [x0, y0] o None para auto-calibrar al arrancar

# ─── CALIBRACIÓN FUERZA-VISIÓN ────────────────────────────────────────────────
# Coeficientes de la regresión lineal: F = K * δ + b
# Generados por force_vision_curve.py
CALIB_K = 0.045    # N/px  (REEMPLAZAR con tu calibración real)
CALIB_B = 0.02     # N     (offset)

# ─── CONTROLADOR PID ─────────────────────────────────────────────────────────
PID_KP = 2.5       # Ganancia proporcional
PID_KI = 0.1       # Ganancia integral
PID_KD = 0.8       # Ganancia derivativa (clave para suavidad)

# Setpoint de fuerza deseada [N]
FORCE_SETPOINT_N  = 0.5   # Fuerza objetivo de agarre suave

# Umbrales de seguridad
DELTA_THRESHOLD_STOP_PX  = 80    # Deflexión máxima: detiene el motor (px)
DELTA_THRESHOLD_WARN_PX  = 50    # Deflexión de advertencia: reduce velocidad (px)

# ─── MOTOR DC / PWM ───────────────────────────────────────────────────────────
MOTOR_PIN_PWM    = 18    # GPIO BCM para señal PWM (Raspberry Pi)
MOTOR_PIN_DIR1   = 23    # GPIO para dirección A
MOTOR_PIN_DIR2   = 24    # GPIO para dirección B
PWM_FREQUENCY    = 1000  # Hz

PWM_MIN          = 0     # 0% — motor detenido
PWM_MAX          = 100   # 100% — velocidad máxima
PWM_INITIAL      = 60    # % — velocidad de cierre inicial

# ─── LOGGER ───────────────────────────────────────────────────────────────────
LOG_DIR          = "data/"
LOG_INTERVAL_MS  = 10    # Guardar cada N ms (100 Hz de logging)

# ─── MODO SIMULACIÓN (sin hardware) ───────────────────────────────────────────
# Poner en True para correr el algoritmo sin GPIO ni cámara real
SIMULATION_MODE  = False
