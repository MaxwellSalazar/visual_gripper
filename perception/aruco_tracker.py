"""
========================================================================
  perception/aruco_tracker.py
  Stage A — Perception: ArUco Marker Detection (optional / higher accuracy)
========================================================================
  Uses OpenCV's ArUco module to detect a fiducial marker printed on the
  gripper finger.  ArUco provides sub-pixel centroid precision and is
  robust to partial occlusion — ideal for peer-reviewed accuracy claims.

  Print marker from:  https://chev.me/arucogen/
  Recommended: DICT_4X4_50, marker ID 0, 30mm physical size.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import ARUCO_DICT, ARUCO_MARKER_ID, ROI


# Map string name → cv2 aruco constant
_ARUCO_DICTS = {
    "DICT_4X4_50":   cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100":  cv2.aruco.DICT_4X4_100,
    "DICT_6X6_50":   cv2.aruco.DICT_6X6_50,
    "DICT_6X6_250":  cv2.aruco.DICT_6X6_250,
}


@dataclass
class ArucoObservation:
    """Result from one ArUco detection."""
    centroid: Tuple[float, float]         # (px, py) in ROI coordinates
    centroid_global: Tuple[float, float]  # (px, py) in full-frame coords
    corners: np.ndarray                   # 4 corner points (1,4,2) float32
    marker_id: int


class ArucoMarkerTracker:
    """
    Detects a single ArUco marker on the gripper finger.
    Falls back gracefully (returns None) when marker is out of view.

    Advantages over color tracker:
    - Sub-pixel centroid accuracy
    - Immune to lighting / color changes
    - Can also estimate 3D pose if camera is calibrated

    Parameters
    ----------
    dict_name   : str — ArUco dictionary key (see _ARUCO_DICTS)
    marker_id   : int — Expected marker ID; None = accept any
    roi         : tuple — (x0, y0, x1, y1) crop region
    """

    def __init__(
        self,
        dict_name: str = ARUCO_DICT,
        marker_id: Optional[int] = ARUCO_MARKER_ID,
        roi: tuple = ROI,
    ):
        aruco_dict_id = _ARUCO_DICTS.get(dict_name, cv2.aruco.DICT_4X4_50)
        self.aruco_dict   = cv2.aruco.getPredefinedDictionary(aruco_dict_id)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector     = cv2.aruco.ArucoDetector(
            self.aruco_dict, self.aruco_params
        )
        self.marker_id    = marker_id
        self.roi          = roi

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> Optional[ArucoObservation]:
        """
        Detect ArUco marker in one BGR frame.

        Returns
        -------
        ArucoObservation | None
        """
        x0, y0, x1, y1 = self.roi
        roi_frame = frame[y0:y1, x0:x1]
        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)

        corners_list, ids, _ = self.detector.detectMarkers(gray)

        if ids is None or len(ids) == 0:
            return None

        # Filter by expected marker ID if set
        for corners, mid in zip(corners_list, ids.flatten()):
            if self.marker_id is not None and mid != self.marker_id:
                continue

            # Centroid = mean of 4 corners
            pts = corners[0]                    # shape (4, 2)
            cx_roi = float(pts[:, 0].mean())
            cy_roi = float(pts[:, 1].mean())
            cx_global = cx_roi + x0
            cy_global = cy_roi + y0

            return ArucoObservation(
                centroid=(cx_roi, cy_roi),
                centroid_global=(cx_global, cy_global),
                corners=corners,
                marker_id=int(mid),
            )

        return None  # No matching marker found

    def draw_overlay(
        self,
        frame: np.ndarray,
        obs: Optional[ArucoObservation],
        deflection_px: float = 0.0,
        force_N: float = 0.0,
    ) -> np.ndarray:
        """Draw ArUco corners and centroid on full frame."""
        out = frame.copy()
        x0, y0, x1, y1 = self.roi
        cv2.rectangle(out, (x0, y0), (x1, y1), (200, 200, 0), 1)

        if obs is None:
            cv2.putText(out, "ArUco: NOT FOUND", (x0, y0 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            return out

        gx, gy = int(obs.centroid_global[0]), int(obs.centroid_global[1])

        # Draw corners (offset by ROI origin)
        for pt in obs.corners[0]:
            cv2.circle(out, (int(pt[0]) + x0, int(pt[1]) + y0),
                       3, (255, 100, 0), -1)

        # Centroid
        cv2.circle(out, (gx, gy), 6, (0, 255, 0), -1)
        cv2.circle(out, (gx, gy), 12, (0, 200, 0), 2)

        info = [
            f"ArUco ID: {obs.marker_id}",
            f"Centroid: ({gx}, {gy}) px",
            f"Deflection: {deflection_px:.1f} px",
            f"Force: {force_N:.3f} N",
        ]
        for i, line in enumerate(info):
            cv2.putText(out, line, (x0, y0 - 8 - i * 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        return out
