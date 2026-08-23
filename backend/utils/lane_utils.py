"""
lane_utils.py — Adaptive horizon-relative ROI
═══════════════════════════════════════════════════════════════════════

WHY THE TRAPEZOID WAS REPLACED
────────────────────────────────
A fixed trapezoid assumes:
  • Horizon is always at the same height     — wrong on hills
  • Road curves fit within fixed side edges  — wrong on bends
  • Lane width is constant                   — wrong on narrow/wide roads

Result: vehicles on curves escape the right edge, vehicles on hills
appear above the top edge — both cause misses.

THE NEW APPROACH: HORIZON-RELATIVE DYNAMIC ZONE
─────────────────────────────────────────────────
Every frame:
  1. Estimate the vanishing point (horizon) from optical flow vectors.
     Flow lines from a moving dashcam converge at the vanishing point.
     This naturally tracks hills (vp moves up) and curves (vp shifts sideways).

  2. Active zone = full frame width, from (vp_y + MARGIN) to frame bottom.
     Full width means no vehicle escapes on curves.
     Horizon-relative means hills don't cause misses.

  3. Lane danger zone = right half of active zone (existing logic in
     process_frame.py — unchanged).

  4. inside_roi() checks a vehicle centroid against this dynamic zone,
     not a hardcoded trapezoid.

FALLBACK
─────────
If vanishing point estimation fails (e.g. ego stopped, no flow):
  Falls back to a fixed fraction of frame height (VP_FALLBACK_FRAC).
  This is more robust than the old trapezoid because it's still full-width.

CPU COST
─────────
Optical flow for vp estimation is already being computed in process_frame.
To avoid double computation, we cache the last known vp and reuse it
if the current frame's estimation fails.
"""

import cv2
import numpy as np
from collections import deque
from utils.draw import blend_rect

# ══════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════

# Fraction of frame height added below estimated vp as a margin.
# Prevents the zone from starting exactly at the vanishing point
# (where perspective distortion is highest and detections unreliable).
VP_MARGIN_FRAC   = 0.08   # zone starts this far below vp

# Fallback: if vp estimation fails, zone starts at this fraction from top.
# 0.45 = zone covers bottom 55% of frame.
VP_FALLBACK_FRAC = 0.45

# EMA smoothing for vp_y — prevents jitter from frame-to-frame variation.
VP_EMA_ALPHA     = 0.15   # low = very smooth, slow to react
                           # raise to 0.30 on hilly roads for faster tracking

# Optical flow params for vp estimation
VP_MAX_CORNERS   = 150
VP_QUALITY       = 0.01
VP_MIN_DIST      = 8

# Colour for ROI overlay
ROI_COLOUR       = (255, 220, 0)    # yellow
ROI_ALPHA        = 0.06             # overlay transparency


# ══════════════════════════════════════════════════════════
#  STATE
# ══════════════════════════════════════════════════════════
_vp_y_smooth  = None     # smoothed vanishing point y (px)
_prev_gray_vp = None     # cached greyscale for vp estimation


# ══════════════════════════════════════════════════════════
#  VANISHING POINT ESTIMATION
# ══════════════════════════════════════════════════════════
def _estimate_vp_y(frame_gray):
    """
    Estimate the vertical position of the vanishing point using
    optical flow direction convergence.

    When a dashcam moves forward, all optical flow vectors point
    away from the vanishing point. The median vertical component
    of background flow vectors converges toward vp_y.

    Returns vp_y in pixels, or None if estimation fails.
    """
    global _prev_gray_vp, _vp_y_smooth

    curr = frame_gray
    if _prev_gray_vp is None:
        _prev_gray_vp = curr
        return _vp_y_smooth
    
    prev = _prev_gray_vp

    p0 = cv2.goodFeaturesToTrack(
        prev, VP_MAX_CORNERS, VP_QUALITY, VP_MIN_DIST)
    _prev_gray_vp = curr

    if p0 is None:
     _prev_gray_vp = curr
     return _vp_y_smooth

    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, curr, p0, None)
    
    _prev_gray_vp = curr
   
    p0g = p0[st == 1]
    p1g = p1[st == 1]
    if len(p0g) < 10:
        return _vp_y_smooth

    # Use RANSAC homography to get background-only points
    H, mask = cv2.findHomography(p0g, p1g, cv2.RANSAC, 3.0)
    if mask is not None:
        inl = mask.ravel().astype(bool)
        if inl.sum() >= 8:
            p0g = p0g[inl]
            p1g = p1g[inl]

    # Flow vectors: direction from which they diverge = vanishing point
    # For forward motion, flow vectors point away from vp.
    # Rays: from p0 in direction (p0 - p1) extended backward.
    # We estimate vp_y as the y-coordinate where most rays converge.
    # Simplified: median of (p0_y - dy * t) where t brings ray to centre_x
    h, w = frame_gray.shape[:2]
    cx = w / 2.0

    vp_ys = []
    for (x0, y0), (x1, y1) in zip(p0g.reshape(-1, 2), p1g.reshape(-1, 2)):
        dx = x0 - x1   # flow direction inverted = toward vp
        dy = y0 - y1
        if abs(dx) < 0.5:
            continue
        # t to reach centre_x: x0 + dx*t = cx → t = (cx - x0) / dx
        t = (cx - x0) / dx
        if t < 0 or t > 50:   # reject diverging or very far intersections
            continue
        vp_y = y0 + dy * t
        if 0 < vp_y < h * 0.75:   # vp must be in upper 3/4 of frame
            vp_ys.append(vp_y)

    if len(vp_ys) < 5:
        return _vp_y_smooth

    raw_vp_y = float(np.median(vp_ys))

    # EMA smooth
    if _vp_y_smooth is None:
        _vp_y_smooth = raw_vp_y
    else:
        _vp_y_smooth = (VP_EMA_ALPHA * raw_vp_y
                        + (1 - VP_EMA_ALPHA) * _vp_y_smooth)

    return _vp_y_smooth


# ══════════════════════════════════════════════════════════
#  PUBLIC API
# ══════════════════════════════════════════════════════════

def get_zone_top_y(frame):
    """
    Returns the y-coordinate of the top edge of the active zone.
    Everything BELOW this y is in the active zone.

    Vehicles above this line are near/above the horizon — too far away
    or in sky — and should be ignored.
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    vp_y = _estimate_vp_y(gray)

    if vp_y is None:
        # Fallback: fixed fraction
        return int(h * VP_FALLBACK_FRAC)

    zone_top = int(vp_y + VP_MARGIN_FRAC * h)
    # Clamp: zone must cover at least bottom 30% of frame
    zone_top = min(zone_top, int(h * 0.70))
    # Zone must start below top 20% (ignore near-sky regions)
    zone_top = max(zone_top, int(h * 0.20))

    return zone_top


def inside_roi(cx, cy, frame):
    """
    Returns True if (cx, cy) is inside the active detection zone.

    Replaces the old fixed trapezoid inside_trapezoid().
    Full frame width — no side cutoff on curves.
    Horizon-adaptive top edge — no misses on hills.
    """
    zone_top = get_zone_top_y(frame)
    h, w     = frame.shape[:2]
    return (zone_top <= cy <= h) and (0 <= cx <= w)


def get_lane(cx, frame):
    """
    Returns 'left', 'right', or 'centre' based on horizontal position.
    Unchanged from previous version.
    """
    h, w = frame.shape[:2]
    if cx < w * 0.45:
        return "left"
    if cx > w * 0.55:
        return "right"
    return "centre"


def get_lane_direction(lane):
    """Expected travel direction for a given lane."""
    directions = {
        "left":   "down",
        "right":  "up",
        "centre": None,
    }
    return directions.get(lane)


def draw_roi(frame):
    """
    Draws the adaptive active zone onto the frame.
    Replaces the old fixed trapezoid draw_roi().

    Shows:
      • Horizontal line at zone top (horizon estimate)
      • Faint overlay tinting the active zone
      • 'active zone' label
    """
    h, w     = frame.shape[:2]
    zone_top = get_zone_top_y(frame)

    # Faint overlay on active zone
    blend_rect(frame, 0, zone_top, w, h, ROI_COLOUR, ROI_ALPHA)

    # Zone top line
    cv2.line(frame, (0, zone_top), (w, zone_top),
             ROI_COLOUR, 1)

    # Label
    cv2.putText(frame, f"active zone (horizon ~{zone_top}px)",
                (8, zone_top - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, ROI_COLOUR, 1)

    return frame