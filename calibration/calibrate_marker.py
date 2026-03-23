"""
calibration/calibrate_marker.py — Asistente interactivo para calibrar el marcador HSV

Cómo usar:
    python calibration/calibrate_marker.py

Controles:
    Trackbars H/S/V (Min y Max): Ajusta el rango del marcador en tiempo real
    's'  → Guarda los valores a config.py automáticamente
    'r'  → Reinicia los valores al estado de config.py
    'q'  → Salir sin guardar

Genera: Un bloque de texto listo para copiar en config.py
"""

import cv2
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


# ── Trackbar callback (OpenCV la requiere, no hace nada) ─────────────────────
def nothing(_): pass


def run_calibration():
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("ERROR: No se pudo abrir la cámara.")
        return

    # ── Ventana de trackbars ──────────────────────────────────────────────────
    win = "HSV Calibration — Marcador del Gripper"
    cv2.namedWindow(win)

    lo = config.MARKER_HSV_LOWER
    hi = config.MARKER_HSV_UPPER

    cv2.createTrackbar("H Min", win, lo[0], 179, nothing)
    cv2.createTrackbar("S Min", win, lo[1], 255, nothing)
    cv2.createTrackbar("V Min", win, lo[2], 255, nothing)
    cv2.createTrackbar("H Max", win, hi[0], 179, nothing)
    cv2.createTrackbar("S Max", win, hi[1], 255, nothing)
    cv2.createTrackbar("V Max", win, hi[2], 255, nothing)

    x, y, w, h = config.ROI
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    print("\n[Calibración] Ajusta los trackbars hasta aislar SOLO el marcador.")
    print("  's' → guardar  |  'r' → reset  |  'q' → salir\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Leer valores actuales
        h_min = cv2.getTrackbarPos("H Min", win)
        s_min = cv2.getTrackbarPos("S Min", win)
        v_min = cv2.getTrackbarPos("V Min", win)
        h_max = cv2.getTrackbarPos("H Max", win)
        s_max = cv2.getTrackbarPos("S Max", win)
        v_max = cv2.getTrackbarPos("V Max", win)

        lower = np.array([h_min, s_min, v_min], dtype=np.uint8)
        upper = np.array([h_max, s_max, v_max], dtype=np.uint8)

        # Aplicar en ROI
        roi_crop = frame[y:y+h, x:x+w]
        hsv      = cv2.cvtColor(roi_crop, cv2.COLOR_BGR2HSV)
        mask     = cv2.inRange(hsv, lower, upper)
        mask     = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        mask     = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Resultado visual
        result   = cv2.bitwise_and(roi_crop, roi_crop, mask=mask)
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # Detectar centroide
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        centroid_txt = "Sin detección"
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) >= config.MARKER_MIN_AREA:
                M = cv2.moments(largest)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.circle(result, (cx, cy), 6, (0, 0, 255), -1)
                    centroid_txt = f"Centroide: ({cx+x}, {cy+y})"

        # HUD con valores actuales
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 255, 255), 2)
        cv2.putText(overlay, centroid_txt, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.putText(overlay,
                    f"HSV Lower: [{h_min}, {s_min}, {v_min}]",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)
        cv2.putText(overlay,
                    f"HSV Upper: [{h_max}, {s_max}, {v_max}]",
                    (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)
        cv2.putText(overlay, "'s'=guardar  'q'=salir",
                    (10, config.CAMERA_HEIGHT - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Mostrar 3 paneles: frame original | máscara | resultado filtrado
        top_row    = np.hstack([overlay, overlay])   # placeholder izquierda
        panel_mask = cv2.resize(mask_bgr, (w, h))
        panel_res  = cv2.resize(result,   (w, h))

        # Layout: frame principal arriba, mini paneles abajo
        bottom     = np.hstack([panel_mask, panel_res,
                                 np.zeros((h, config.CAMERA_WIDTH - 2*w, 3),
                                          dtype=np.uint8)])
        display    = np.vstack([overlay, bottom])
        cv2.imshow(win, display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            _save_calibration(h_min, s_min, v_min, h_max, s_max, v_max)
        elif key == ord('r'):
            cv2.setTrackbarPos("H Min", win, config.MARKER_HSV_LOWER[0])
            cv2.setTrackbarPos("S Min", win, config.MARKER_HSV_LOWER[1])
            cv2.setTrackbarPos("V Min", win, config.MARKER_HSV_LOWER[2])
            cv2.setTrackbarPos("H Max", win, config.MARKER_HSV_UPPER[0])
            cv2.setTrackbarPos("S Max", win, config.MARKER_HSV_UPPER[1])
            cv2.setTrackbarPos("V Max", win, config.MARKER_HSV_UPPER[2])
            print("[Calibración] Reset a valores de config.py")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def _save_calibration(h_min, s_min, v_min, h_max, s_max, v_max):
    """Muestra los valores listos para copiar en config.py."""
    print("\n" + "="*55)
    print("VALORES CALIBRADOS — copia en config.py:")
    print(f"  MARKER_HSV_LOWER = [{h_min}, {s_min}, {v_min}]")
    print(f"  MARKER_HSV_UPPER = [{h_max}, {s_max}, {v_max}]")
    print("="*55)

    # También guarda en archivo para referencia
    os.makedirs("data", exist_ok=True)
    with open("data/marker_hsv.txt", "w") as f:
        f.write(f"MARKER_HSV_LOWER = [{h_min}, {s_min}, {v_min}]\n")
        f.write(f"MARKER_HSV_UPPER = [{h_max}, {s_max}, {v_max}]\n")
    print("[Calibración] Guardado en data/marker_hsv.txt")


if __name__ == "__main__":
    run_calibration()
