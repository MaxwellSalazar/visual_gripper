"""
========================================================================
  perception/color_tracker.py
  Stage A — Perception: Color Marker Detection
========================================================================
  Detects the centroid of a colored marker placed on the flexible finger.
  Returns (px, py) in image coordinates, or None if marker not found.

  Technique: HSV thresholding → morphological cleanup → largest contour
  centroid. Robust to mild lighting changes; tune HSV bounds via
  calibration/color_picker.py before running the main experiment.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import (
    MARKER_COLOR_LOWER, MARKER_COLOR_UPPER,
    MARKER_MIN_AREA, ROI
)


@dataclass
class MarkerObservation:
    """All visual information extracted from one frame."""
    centroid: Tuple[float, float]   # (px, py) in ROI-local coordinates
    centroid_global: Tuple[float, float]  # (px, py) in full-frame coords
    area: float                     # contour area in px²
    mask: np.ndarray                # binary mask (for debug overlay)
    contour: np.ndarray             # raw contour points


class ColorMarkerTracker:
    """
    Detects a single colored marker on the gripper finger using HSV
    thresholding.  Designed for real-time operation at 60 fps on a
    Raspberry Pi 4 or equivalent.

    Parameters
    ----------
    lower_hsv : list[int]  — [H, S, V] lower bound
    upper_hsv : list[int]  — [H, S, V] upper bound
    min_area  : float      — minimum contour area to consider valid
    roi       : tuple      — (x0, y0, x1, y1) region of interest in px
    """

    def __init__(
        self,
        lower_hsv: list = MARKER_COLOR_LOWER,
        upper_hsv: list = MARKER_COLOR_UPPER,
        min_area: float = MARKER_MIN_AREA,
        roi: tuple = ROI,
    ):
        self.lower = np.array(lower_hsv, dtype=np.uint8)
        self.upper = np.array(upper_hsv, dtype=np.uint8)
        self.min_area = min_area
        self.roi = roi                        # (x0, y0, x1, y1)

        # Morphological kernels — remove noise while preserving marker blob
        self._kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self._kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> Optional[MarkerObservation]:
        """
        Process one BGR frame and return a MarkerObservation, or None
        if the marker is not visible or the blob is too small.

        Parameters
        ----------
        frame : np.ndarray  — BGR image from cv2.VideoCapture

        Returns
        -------
        MarkerObservation | None
        """
        roi_frame = self._crop_roi(frame)

        # 1. Convert to HSV — more robust to illumination than RGB
        hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)

        # 2. Threshold — isolate marker pixels
        mask = cv2.inRange(hsv, self.lower, self.upper)

        # 3. Morphological cleanup
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self._kernel_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel_close)

        # 4. Find contours — pick the largest valid blob
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None

        best = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(best)
        if area < self.min_area:
            return None

        # 5. Compute centroid via image moments
        M = cv2.moments(best)
        if M["m00"] == 0:
            return None

        cx_roi = M["m10"] / M["m00"]
        cy_roi = M["m01"] / M["m00"]

        # Convert ROI-local → global frame coordinates
        x0, y0, _, _ = self.roi
        cx_global = cx_roi + x0
        cy_global = cy_roi + y0

        return MarkerObservation(
            centroid=(cx_roi, cy_roi),
            centroid_global=(cx_global, cy_global),
            area=area,
            mask=mask,
            contour=best,
        )

    def draw_overlay(
        self,
        frame: np.ndarray,
        obs: Optional[MarkerObservation],
        deflection_px: float = 0.0,
        force_N: float = 0.0,
    ) -> np.ndarray:
        """
        Draw debug overlay on the full frame.  Call after detect().
        Returns the annotated frame (does NOT modify in-place).
        """
        out = frame.copy()
        x0, y0, x1, y1 = self.roi

        # ROI rectangle
        cv2.rectangle(out, (x0, y0), (x1, y1), (200, 200, 0), 1)

        if obs is None:
            cv2.putText(out, "MARKER: NOT FOUND", (x0, y0 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            return out

        gx, gy = int(obs.centroid_global[0]), int(obs.centroid_global[1])

        # Detected centroid
        cv2.circle(out, (gx, gy), 6, (0, 255, 0), -1)
        cv2.circle(out, (gx, gy), 10, (0, 200, 0), 2)

        # Info text
        info = [
            f"Centroid: ({gx}, {gy}) px",
            f"Deflection: {deflection_px:.1f} px",
            f"Force: {force_N:.3f} N",
            f"Area: {obs.area:.0f} px2",
        ]
        for i, line in enumerate(info):
            cv2.putText(out, line, (x0, y0 - 8 - i * 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        return out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _crop_roi(self, frame: np.ndarray) -> np.ndarray:
        x0, y0, x1, y1 = self.roi
        return frame[y0:y1, x0:x1]
