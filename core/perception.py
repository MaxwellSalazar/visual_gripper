"""
core/perception.py — Etapa A del Pipeline: Percepción Visual

Responsabilidades:
  1. Capturar frames de la cámara a alta velocidad
  2. Aislar el marcador del dedo flexible usando filtrado HSV dentro de la ROI
  3. Calcular el centroide del marcador → (Px, Py) en píxeles
  4. Retornar la posición del marcador y una imagen anotada para visualización

Diseño: clase stateless — puede llamarse frame a frame sin acumular estado.
"""

import cv2
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


class MarkerPerception:
    """
    Detecta el centroide de un marcador de color en la imagen del gripper.

    Parámetros principales (de config.py):
        ROI              : Región de Interés [x, y, w, h]
        MARKER_HSV_LOWER : Límite inferior del rango HSV del marcador
        MARKER_HSV_UPPER : Límite superior del rango HSV del marcador
        MARKER_MIN_AREA  : Área mínima del blob válido (filtra ruido)
    """

    def __init__(self):
        self.hsv_lower = np.array(config.MARKER_HSV_LOWER, dtype=np.uint8)
        self.hsv_upper = np.array(config.MARKER_HSV_UPPER, dtype=np.uint8)
        self.roi       = config.ROI          # [x, y, w, h]
        self.min_area  = config.MARKER_MIN_AREA

        # Kernel morfológico para cerrar huecos en la máscara
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def process(self, frame: np.ndarray):
        """
        Procesa un frame BGR y devuelve la posición del centroide.

        Args:
            frame : Imagen BGR de OpenCV (H x W x 3)

        Returns:
            centroid : (cx, cy) en coordenadas del frame completo,
                       o None si no se detectó el marcador.
            debug_img: Frame anotado con ROI, máscara y centroide.
            mask     : Máscara binaria de la ROI (para análisis posterior).
        """
        debug_img = frame.copy()
        x, y, w, h = self.roi

        # ── 1. Recortar ROI ──────────────────────────────────────────────────
        roi_crop = frame[y:y+h, x:x+w]

        # ── 2. Filtrado HSV ──────────────────────────────────────────────────
        hsv  = cv2.cvtColor(roi_crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)

        # Operaciones morfológicas: elimina ruido y cierra el blob
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self.morph_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.morph_kernel)

        # ── 3. Encontrar contornos y elegir el blob más grande ───────────────
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        centroid = None
        if contours:
            largest = max(contours, key=cv2.contourArea)
            area    = cv2.contourArea(largest)

            if area >= self.min_area:
                M  = cv2.moments(largest)
                if M["m00"] > 0:
                    # Centroide en coordenadas de la ROI
                    cx_roi = int(M["m10"] / M["m00"])
                    cy_roi = int(M["m01"] / M["m00"])

                    # Convertir a coordenadas del frame completo
                    cx = cx_roi + x
                    cy = cy_roi + y
                    centroid = (cx, cy)

                    # ── Anotaciones en debug_img ─────────────────────────
                    cv2.drawContours(debug_img[y:y+h, x:x+w],
                                     [largest], -1, (0, 255, 0), 2)
                    cv2.circle(debug_img, (cx, cy), 6, (0, 0, 255), -1)
                    cv2.putText(debug_img,
                                f"({cx},{cy})",
                                (cx + 8, cy - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                (0, 0, 255), 1)

        # Dibuja la ROI siempre
        color_roi = (0, 255, 255) if centroid else (0, 100, 200)
        cv2.rectangle(debug_img, (x, y), (x+w, y+h), color_roi, 2)
        cv2.putText(debug_img, "ROI", (x + 4, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_roi, 1)

        return centroid, debug_img, mask


class CameraStream:
    """
    Maneja la captura de video a alta velocidad desde la cámara.
    Usa un buffer de 1 frame para minimizar latencia (no acumula frames viejos).
    """

    def __init__(self):
        idx = config.CAMERA_INDEX
        self.cap = cv2.VideoCapture(idx)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS,          config.CAMERA_FPS)
        # Buffer mínimo: reduce latencia de captura
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            raise RuntimeError(f"No se pudo abrir la cámara (índice {idx}). "
                               "Verifica la conexión o cambia CAMERA_INDEX en config.py")

        # Medir FPS real alcanzado
        self._fps_real = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"[CameraStream] Cámara abierta — FPS configurados: {config.CAMERA_FPS} "
              f"| FPS reales: {self._fps_real:.1f}")

    def read(self):
        """Lee el frame más reciente. Retorna (ok, frame)."""
        return self.cap.read()

    def release(self):
        self.cap.release()

    @property
    def fps(self):
        return self._fps_real


def measure_actual_fps(camera: CameraStream, n_frames: int = 120) -> float:
    """
    Mide el FPS real del pipeline de captura.
    Útil para la sección de latencia en la publicación.

    Returns:
        fps_measured : FPS promedio medido empíricamente.
    """
    start = time.perf_counter()
    for _ in range(n_frames):
        ok, _ = camera.read()
        if not ok:
            break
    elapsed = time.perf_counter() - start
    fps_measured = n_frames / elapsed
    print(f"[FPS] {n_frames} frames en {elapsed:.3f}s → {fps_measured:.2f} FPS reales")
    return fps_measured
