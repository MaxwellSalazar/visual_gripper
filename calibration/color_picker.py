"""
========================================================================
  calibration/color_picker.py
  Interactive HSV Tuner — Find your marker's HSV range
========================================================================
  Run this script and adjust the trackbars until the binary mask
  shows ONLY the marker blob.  Then copy the H/S/V values printed to
  the console into config/settings.py.

  Usage:
    python calibration/color_picker.py
"""

import cv2
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.camera import Camera


def run_color_picker():
    print("=== HSV Color Picker ===")
    print("Adjust trackbars until ONLY your marker appears white in the mask.")
    print("Press 'p' to print current values.  Press 'q' to quit.\n")

    win_name   = "HSV Tuner"
    mask_name  = "Mask"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.namedWindow(mask_name, cv2.WINDOW_NORMAL)

    # Trackbars: H [0–179], S [0–255], V [0–255]
    def nothing(_): pass
    cv2.createTrackbar("H Low",  win_name,  35, 179, nothing)
    cv2.createTrackbar("H High", win_name,  85, 179, nothing)
    cv2.createTrackbar("S Low",  win_name, 100, 255, nothing)
    cv2.createTrackbar("S High", win_name, 255, 255, nothing)
    cv2.createTrackbar("V Low",  win_name, 100, 255, nothing)
    cv2.createTrackbar("V High", win_name, 255, 255, nothing)

    with Camera() as cam:
        while True:
            frame, _ = cam.read()
            if frame is None:
                break

            h_lo = cv2.getTrackbarPos("H Low",  win_name)
            h_hi = cv2.getTrackbarPos("H High", win_name)
            s_lo = cv2.getTrackbarPos("S Low",  win_name)
            s_hi = cv2.getTrackbarPos("S High", win_name)
            v_lo = cv2.getTrackbarPos("V Low",  win_name)
            v_hi = cv2.getTrackbarPos("V High", win_name)

            lower = np.array([h_lo, s_lo, v_lo], dtype=np.uint8)
            upper = np.array([h_hi, s_hi, v_hi], dtype=np.uint8)

            hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower, upper)

            # Morphological cleanup to remove noise
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            # Overlay contours on original frame
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            display = frame.copy()
            cv2.drawContours(display, contours, -1, (0, 255, 0), 2)

            cv2.imshow(win_name,  display)
            cv2.imshow(mask_name, mask)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('p'):
                print(f"\n--- Copy these values to config/settings.py ---")
                print(f"MARKER_COLOR_LOWER = [{h_lo}, {s_lo}, {v_lo}]")
                print(f"MARKER_COLOR_UPPER = [{h_hi}, {s_hi}, {v_hi}]")
                print("------------------------------------------------\n")
            elif key == ord('q'):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_color_picker()
