"""
========================================================================
  Visual Gripper - Configuration Settings
  Adaptive Grasp Control via Proprioceptive Visual Servoing
========================================================================
  Modify these parameters to match your physical hardware setup.
  All tuneable values are centralized here — never hardcode in modules.
"""

# ── Camera ─────────────────────────────────────────────────────────────
CAMERA_INDEX          = 0          # 0 = first USB webcam
CAMERA_WIDTH          = 640        # pixels
CAMERA_HEIGHT         = 480        # pixels
CAMERA_FPS            = 60         # target fps; driver/HW may cap this

# ── Motor / PWM (L298N via Arduino serial) ─────────────────────────────
SERIAL_PORT           = "COM3"           # Windows: COM3, COM4... | Linux: /dev/ttyUSB0
SERIAL_BAUD           = 115200
PWM_MAX               = 255        # 8-bit PWM ceiling
PWM_MIN               = 0
PWM_IDLE              = 0          # motor stopped
PWM_OPEN              = 200        # speed to open the gripper

# ── PID Controller ─────────────────────────────────────────────────────
PID_KP                = 2.5        # proportional gain
PID_KI                = 0.1        # integral gain
PID_KD                = 0.8        # derivative gain
PID_SETPOINT          = 0.0        # target deflection (px) — 0 = no contact
PID_OUTPUT_LIMITS     = (0, PWM_MAX)

# ── Deflection Thresholds ───────────────────────────────────────────────
DEFLECTION_CONTACT    = 5.0        # px — first contact detected
DEFLECTION_SOFT_STOP  = 25.0       # px — reduce speed (fragile objects)
DEFLECTION_HARD_STOP  = 45.0       # px — full stop (max safe deformation)

# ── Calibration (Force ↔ Deflection) ───────────────────────────────────
# F(N) = CALIB_SLOPE * delta(px) + CALIB_INTERCEPT
# These defaults are placeholders — run calibration/calibrate.py first
CALIB_SLOPE           = 0.052      # N/px  (example value)
CALIB_INTERCEPT       = -0.15      # N     (example value)

# ── Color Marker (HSV bounds) ───────────────────────────────────────────
# Use calibration/color_picker.py to find your marker's HSV range
MARKER_COLOR_LOWER    = [35, 100, 100]   # H, S, V  — green marker example
MARKER_COLOR_UPPER    = [85, 255, 255]
MARKER_MIN_AREA       = 50              # px² — ignore tiny noise blobs

# ── ArUco Marker ────────────────────────────────────────────────────────
ARUCO_DICT            = "DICT_4X4_50"   # must match printed marker
ARUCO_MARKER_ID       = 0

# ── Region of Interest (ROI) ────────────────────────────────────────────
# Crop to the finger area to reduce processing load and false detections
# (x_start, y_start, x_end, y_end) in pixels
ROI                   = (150, 80, 490, 400)

# ── Data Logging ────────────────────────────────────────────────────────
LOG_DIR               = "data/logs"
LOG_ENABLED           = True
LOG_COLUMNS           = ["timestamp", "frame_id", "px", "py",
                          "deflection_px", "force_N", "pwm", "state"]

# ── Display ─────────────────────────────────────────────────────────────
DISPLAY_ENABLED       = True       # set False on headless Raspberry Pi
DISPLAY_SCALE         = 1.0        # scale factor for the debug window
