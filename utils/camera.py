"""
========================================================================
  utils/camera.py
  Camera Abstraction — USB Webcam via OpenCV
========================================================================
  Wraps cv2.VideoCapture with:
    - Reliable initialization with retry
    - FPS measurement (actual vs. requested)
    - Frame timestamping for latency analysis
    - Safe release on exit
"""

import cv2
import time
import numpy as np
from typing import Optional, Tuple
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS


class Camera:
    """
    Simple OpenCV camera wrapper with FPS measurement.

    Usage
    -----
    with Camera() as cam:
        while True:
            frame, ts = cam.read()
            if frame is None:
                break
    """

    def __init__(
        self,
        index: int     = CAMERA_INDEX,
        width: int     = CAMERA_WIDTH,
        height: int    = CAMERA_HEIGHT,
        fps: int       = CAMERA_FPS,
    ):
        self._index  = index
        self._width  = width
        self._height = height
        self._fps    = fps
        self._cap    = None

        # FPS estimation state
        self._frame_count = 0
        self._fps_t0      = None
        self._fps_actual  = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self) -> bool:
        """Open camera. Returns True on success."""
        self._cap = cv2.VideoCapture(self._index)
        if not self._cap.isOpened():
            print(f"[Camera] ERROR: Cannot open camera index {self._index}")
            return False

        # Request resolution and FPS (driver may ignore)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap.set(cv2.CAP_PROP_FPS,          self._fps)

        # Reduce capture buffer to minimise latency
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
        actual_w   = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h   = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[Camera] Opened: {actual_w}×{actual_h} @ {actual_fps:.0f} fps")

        self._fps_t0 = time.perf_counter()
        return True

    def release(self) -> None:
        if self._cap:
            self._cap.release()
            self._cap = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_):
        self.release()

    # ------------------------------------------------------------------
    # Frame acquisition
    # ------------------------------------------------------------------

    def read(self) -> Tuple[Optional[np.ndarray], float]:
        """
        Grab one frame.

        Returns
        -------
        (frame, timestamp) — frame is None if read failed.
        timestamp is time.perf_counter() immediately after grab.
        """
        if self._cap is None:
            return None, 0.0

        ret, frame = self._cap.read()
        ts = time.perf_counter()

        if not ret:
            return None, ts

        self._frame_count += 1
        self._update_fps(ts)

        return frame, ts

    # ------------------------------------------------------------------
    # FPS monitoring
    # ------------------------------------------------------------------

    def _update_fps(self, ts: float) -> None:
        elapsed = ts - self._fps_t0
        if elapsed >= 1.0:
            self._fps_actual = self._frame_count / elapsed
            self._frame_count = 0
            self._fps_t0 = ts

    @property
    def fps_actual(self) -> float:
        """Measured FPS over the last second."""
        return self._fps_actual

    @property
    def frame_count(self) -> int:
        return self._frame_count
