"""
motion_utils.py

Optical flow is kept here for future use but returns 0.0 immediately
since camera motion compensation is disabled in the detector.
Removing the Farneback call recovers significant CPU — it was the main
cause of the video running slowly.
"""

import cv2
import numpy as np


def get_camera_motion(
    frame: np.ndarray,
    prev_gray,
):
    """
    Returns (0.0, gray) — compensation disabled, flow not computed.
    Keeping the same signature so main.py needs no changes.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return 0.0, gray