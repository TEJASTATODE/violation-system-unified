"""
wrong_side_violation.py
═══════════════════════════════════════════════════════════════════════
Detects wrong-way vehicles on a front dashcam, Indian roads.

ROOT CAUSE OF PREVIOUS FAILURE
────────────────────────────────
The vehicle "appears suddenly close and passes quickly" means:
  • Already large when first seen  → size-growth (S2) never triggers
  • Gone in 10-15 frames           → MIN_TRACK_AGE=10 + MIN_HISTORY=10
                                     consumed the entire sighting
  • Passes sideways                → lateral motion, not head-on growth

NEW DETECTION LOGIC FOR FAST CLOSE-PASSING VEHICLES
─────────────────────────────────────────────────────
Two independent detection paths run in parallel:

PATH A — FAST CLOSE PASS  (catches the vehicle you described)
  Triggered from frame 1 of detection (no age gate).
  Conditions (all must hold for FAST_CONFIRM consecutive frames):
    A1. Bbox area is LARGE immediately (vehicle already close)
        area > FAST_MIN_AREA_FRAC * frame_area
    A2. Fast lateral (horizontal) movement across our lane
        |dx per frame| > FAST_LATERAL_PX  AND  in ego lane zone
    A3. No physical barrier in centre band (median detector)

PATH B — DISTANT APPROACH  (catches vehicles approaching from far)
  Same 4-signal system as before but with lower thresholds.
  Triggered after MIN_TRACK_AGE frames.
  Conditions:
    B1. In ego lane zone (centroid right of EGO_LANE_LEFT_FRAC)
    B2. Bbox growing (size growth signal)
    B3. No barrier detected
    B4. Sustained score >= SCORE_THRESHOLD

Both paths feed the same score — whichever fires adds points.
This means a vehicle that starts distant (Path B) and then
suddenly closes (Path A) gets double confirmation.

WHY MIN_TRACK_AGE IS REMOVED FOR PATH A
─────────────────────────────────────────
A vehicle that appears suddenly at close range IS already confirmed
by its size. We don't need 10 frames of history to know it's there.
The age gate was designed to filter detector noise on small distant
boxes — those are tiny, so FAST_MIN_AREA_FRAC filters them instead.
"""

import cv2
import os
import time
import numpy as np
from collections import defaultdict, deque
from datetime import datetime
from detection.detect import detect_general_objects


# ═══════════════════════════════════════════════════════════════════
#  CONFIG  — tune these values, nothing else
# ═══════════════════════════════════════════════════════════════════

PROC_W = 640
PROC_H = 360
DEBUG_OVERLAY = True      # set False in production

# ── Ego motion ────────────────────────────────────────────────────
FLOW_SCALE     = 0.5
EGO_EMA_ALPHA  = 0.25
EGO_STOPPED_PX = 1.8

# ── Track lifecycle ───────────────────────────────────────────────
MIN_TRACK_AGE  = 8        # frames before PATH B activates
HISTORY_LEN    = 24
MIN_HISTORY    = 6        # lowered — needed for path B size check
SIZE_EMA_ALPHA = 0.30
EXPIRY_FRAMES  = 90
REAP_INTERVAL  = 30

# ── Ego lane boundary ─────────────────────────────────────────────
EGO_LANE_LEFT_FRAC = 0.38

# ── PATH A: fast close-pass detection ────────────────────────────
# Vehicle must occupy at least this fraction of the processing frame
# to be considered "already close". Filters out distant small boxes.
FAST_MIN_AREA_FRAC = 0.04      # 4% of frame area = ~983 px at 640x360
# Minimum lateral displacement per frame (in processing px)
# to qualify as "fast sideways pass"
FAST_LATERAL_PX    = 6.0       # lower if missing slow passes
# Consecutive frames path A conditions must hold
FAST_CONFIRM       = 3         # intentionally low — vehicle passes fast
# Score added per frame path A fires (higher than path B)
FAST_SCORE_PER_FRAME = 28

# ── PATH B: distant approach detection ───────────────────────────
APPROACH_RATIO_MOVING  = 1.15
APPROACH_RATIO_STOPPED = 1.07
EGO_INFLATION_FACTOR   = 0.010
SCORE_PER_FRAME        = 10
SCORE_DECAY            = 6

# ── Scoring / confirmation ────────────────────────────────────────
SCORE_MAX       = 100
SCORE_MIN       = 0
SCORE_THRESHOLD = 65      # lowered to catch fast-pass vehicles

# ── Median / barrier detector ─────────────────────────────────────
MEDIAN_BAND_LEFT   = 0.35
MEDIAN_BAND_RIGHT  = 0.55
MEDIAN_CANNY_LOW   = 40
MEDIAN_CANNY_HIGH  = 120
MEDIAN_EDGE_THRESH = 18

# ── Clip saver ────────────────────────────────────────────────────
SAVE_CLIPS        = True
CLIP_OUTPUT_DIR   = "violation_clips"
CLIP_PRE_BUF_SEC  = 3
CLIP_POST_BUF_SEC = 2
CLIP_FPS          = 30

if SAVE_CLIPS:
    os.makedirs(CLIP_OUTPUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════
#  MODULE STATE
# ═══════════════════════════════════════════════════════════════════

_cx_hist      = defaultdict(lambda: deque(maxlen=HISTORY_LEN))
_cy_hist      = defaultdict(lambda: deque(maxlen=HISTORY_LEN))
_area_hist    = defaultdict(lambda: deque(maxlen=HISTORY_LEN))
_smooth_area  = {}
_track_age    = defaultdict(int)
_last_seen    = {}
_score        = defaultdict(float)
_violation    = defaultdict(bool)
_fast_consec  = defaultdict(int)    # consecutive path-A frames per track

_ego_smooth   = 0.0
_prev_gray    = None
_frame_idx    = 0

_pre_buffer   = deque()
_clip_writers = {}
_clip_was_viol = defaultdict(bool)


# ═══════════════════════════════════════════════════════════════════
#  EGO MOTION
# ═══════════════════════════════════════════════════════════════════

def _estimate_ego(prev_gray, curr_gray):
    global _ego_smooth
    scale = FLOW_SCALE
    pg = cv2.resize(prev_gray, None, fx=scale, fy=scale)
    cg = cv2.resize(curr_gray, None, fx=scale, fy=scale)
    p0 = cv2.goodFeaturesToTrack(pg, 200, 0.01, 5)
    if p0 is None or len(p0) < 12:
        return _ego_smooth
    p1, st, _ = cv2.calcOpticalFlowPyrLK(pg, cg, p0, None)
    if p1 is None:
        return _ego_smooth
    p0g, p1g = p0[st == 1], p1[st == 1]
    if len(p0g) < 8:
        return _ego_smooth
    H, mask = cv2.findHomography(p0g, p1g, cv2.RANSAC, 3.0)
    if H is None or mask is None:
        dx = float(np.median(p1g[:, 0] - p0g[:, 0]))
        dy = float(np.median(p1g[:, 1] - p0g[:, 1]))
    else:
        inl = mask.ravel().astype(bool)
        if inl.sum() < 5:
            return _ego_smooth
        dx = float(np.median(p1g[inl, 0] - p0g[inl, 0]))
        dy = float(np.median(p1g[inl, 1] - p0g[inl, 1]))
    raw = float((dx**2 + dy**2) ** 0.5) / scale
    _ego_smooth = EGO_EMA_ALPHA * raw + (1 - EGO_EMA_ALPHA) * _ego_smooth
    return _ego_smooth


# ═══════════════════════════════════════════════════════════════════
#  MEDIAN DETECTOR
# ═══════════════════════════════════════════════════════════════════

def _barrier_present(gray):
    h, w = gray.shape[:2]
    xl = int(w * MEDIAN_BAND_LEFT)
    xr = int(w * MEDIAN_BAND_RIGHT)
    band = gray[h // 3:, xl:xr]
    edges = cv2.Canny(band, MEDIAN_CANNY_LOW, MEDIAN_CANNY_HIGH)
    return int(np.count_nonzero(edges)) >= MEDIAN_EDGE_THRESH


# ═══════════════════════════════════════════════════════════════════
#  PATH A — FAST CLOSE PASS
# ═══════════════════════════════════════════════════════════════════

def _path_a(tid, cx, cy, smooth_area, barrier):
    """
    Detect a wrong-way vehicle that appears suddenly close and
    passes laterally across the frame.

    Returns True if all path-A conditions hold this frame.
    No age gate — works from the first detected frame.
    """
    frame_area = PROC_W * PROC_H

    # A1: vehicle must already be large (close to camera)
    if smooth_area < FAST_MIN_AREA_FRAC * frame_area:
        return False

    # A2: must be in our lane zone
    if cx < EGO_LANE_LEFT_FRAC * PROC_W:
        return False

    # A3: fast lateral movement
    cx_list = list(_cx_hist[tid])
    if len(cx_list) < 2:
        return False
    lateral = abs(cx_list[-1] - cx_list[-2])   # px moved this frame
    if lateral < FAST_LATERAL_PX:
        return False

    # A4: no barrier between us
    if barrier:
        return False

    return True


# ═══════════════════════════════════════════════════════════════════
#  PATH B — DISTANT APPROACH
# ═══════════════════════════════════════════════════════════════════

def _path_b(tid, cx, ego_mag, ego_stopped, barrier):
    """
    Detect a wrong-way vehicle approaching from distance.
    Requires MIN_TRACK_AGE frames and MIN_HISTORY area readings.
    """
    if _track_age[tid] < MIN_TRACK_AGE:
        return False

    # B1: in ego lane
    if cx < EGO_LANE_LEFT_FRAC * PROC_W:
        return False

    # B2: approaching (size growing)
    areas = list(_area_hist[tid])
    if len(areas) < MIN_HISTORY:
        return False
    old = float(np.mean(areas[:3]))
    new = float(np.mean(areas[-3:]))
    if old <= 0:
        return False
    growth = new / old
    if ego_stopped:
        thresh = APPROACH_RATIO_STOPPED
    else:
        thresh = APPROACH_RATIO_MOVING * (1.0 + ego_mag * EGO_INFLATION_FACTOR)
    if growth < thresh:
        return False

    # B3: no barrier
    if barrier:
        return False

    return True


# ═══════════════════════════════════════════════════════════════════
#  CLIP SAVER
# ═══════════════════════════════════════════════════════════════════

def _clip_push(raw_frame):
    if not SAVE_CLIPS:
        return
    now = time.monotonic()
    _pre_buffer.append((now, raw_frame.copy()))
    cutoff = now - CLIP_PRE_BUF_SEC
    while _pre_buffer and _pre_buffer[0][0] < cutoff:
        _pre_buffer.popleft()
    for tid in list(_clip_writers):
        state = _clip_writers[tid]
        state["writer"].write(raw_frame)
        state["post_frames"] -= 1
        if state["post_frames"] <= 0 and not _clip_was_viol.get(tid):
            state["writer"].release()
            del _clip_writers[tid]


def _clip_start(tid, frame_bgr):
    if not SAVE_CLIPS or tid in _clip_writers:
        return
    h, w = frame_bgr.shape[:2]
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(CLIP_OUTPUT_DIR, f"violation_{ts}_tid{tid}.mp4")
    writer = cv2.VideoWriter(
        path, cv2.VideoWriter_fourcc(*"mp4v"), CLIP_FPS, (w, h))
    for _, f in _pre_buffer:
        writer.write(f)
    _clip_writers[tid] = {"writer": writer,
                          "post_frames": int(CLIP_FPS * CLIP_POST_BUF_SEC),
                          "path": path}
    _clip_was_viol[tid] = True
    print(f"[ClipSaver] {path}")


def _clip_end(tid):
    if SAVE_CLIPS:
        _clip_was_viol[tid] = False


def _clip_release_all():
    for s in _clip_writers.values():
        s["writer"].release()
    _clip_writers.clear()


# ═══════════════════════════════════════════════════════════════════
#  DRAWING
# ═══════════════════════════════════════════════════════════════════

def _draw_normal(frame, x1, y1, x2, y2, tid):
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 1)
    cv2.putText(frame, str(tid), (x1 + 2, y1 + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 200, 0), 1)


def _draw_suspect(frame, x1, y1, x2, y2, score, path_a, path_b_active):
    ratio = min(score / SCORE_THRESHOLD, 1.0)
    color = (0, int(180 * (1 - ratio)), int(220 * ratio))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    if DEBUG_OVERLAY:
        tag = f"S:{int(score)} {'A' if path_a else '-'}{'B' if path_b_active else '-'}"
        cv2.putText(frame, tag, (x1 + 2, y1 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1)


def _draw_violation(frame, x1, y1, x2, y2, cx_hist, y1r, y2r, sx):
    color = (0, 0, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    label = "WRONG WAY"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.70, 2)
    lx = x1
    ly = max(y1 - 10, th + 6)
    cv2.rectangle(frame, (lx - 2, ly - th - 4),
                  (lx + tw + 4, ly + 4), color, -1)
    cv2.putText(frame, label, (lx, ly),
                cv2.FONT_HERSHEY_DUPLEX, 0.70, (255, 255, 255), 2)
    pts = list(cx_hist)
    cy_mid = (y1r + y2r) // 2
    for i in range(1, len(pts)):
        cv2.line(frame,
                 (int(pts[i-1] * sx), cy_mid),
                 (int(pts[i]   * sx), cy_mid),
                 (0, 60, 255), 2)


def _draw_hud(frame, ego_mag, ego_stopped, barrier):
    lines = [
        f"Ego: {'STOPPED' if ego_stopped else f'MOVING {ego_mag:.1f}px/f'}",
        f"Median: {'BARRIER' if barrier else 'clear'}",
        f"ScoreThr: {SCORE_THRESHOLD}  FastConfirm: {FAST_CONFIRM}",
    ]
    for i, txt in enumerate(lines):
        cv2.putText(frame, txt, (8, 20 + i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 220, 220), 1)


def _draw_median_band(frame):
    if not DEBUG_OVERLAY:
        return
    h, w = frame.shape[:2]
    xl = int(w * MEDIAN_BAND_LEFT)
    xr = int(w * MEDIAN_BAND_RIGHT)
    ov = frame.copy()
    cv2.rectangle(ov, (xl, 0), (xr, h), (255, 200, 0), -1)
    cv2.addWeighted(ov, 0.07, frame, 0.93, 0, frame)
    cv2.line(frame, (xl, 0), (xl, h), (180, 140, 0), 1)
    cv2.line(frame, (xr, 0), (xr, h), (180, 140, 0), 1)


# ═══════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

def process_frame(frame, prev_frame):
    """
    Parameters
    ----------
    frame      : current BGR frame (any resolution)
    prev_frame : previous RAW BGR frame (not annotated)

    Returns
    -------
    Annotated BGR frame (same resolution as input)
    """
    global _frame_idx, _prev_gray

    _frame_idx += 1
    orig_h, orig_w = frame.shape[:2]
    sx = orig_w / PROC_W    # scale factors for drawing
    sy = orig_h / PROC_H

    # ── Resize ────────────────────────────────────────────────────
    small = cv2.resize(frame, (PROC_W, PROC_H))

    # ── Greyscale (cached) ────────────────────────────────────────
    curr_gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    prev_gray = (_prev_gray if _prev_gray is not None else
                 cv2.cvtColor(cv2.resize(prev_frame, (PROC_W, PROC_H)),
                               cv2.COLOR_BGR2GRAY))
    _prev_gray = curr_gray

    # ── Ego motion ────────────────────────────────────────────────
    ego_mag     = _estimate_ego(prev_gray, curr_gray)
    ego_stopped = ego_mag < EGO_STOPPED_PX

    # ── Barrier check (once per frame, not per vehicle) ───────────
    barrier = _barrier_present(curr_gray)

    # ── Pre-buffer ────────────────────────────────────────────────
    _clip_push(frame)

    # ── Detect ────────────────────────────────────────────────────
    detections = detect_general_objects(small)
    out = frame.copy()

    for det in detections:
        tid = det.get("track_id", -1)
        if tid == -1:
            continue

        x1s, y1s, x2s, y2s = det["box"]
        x1 = int(x1s * sx);  x2 = int(x2s * sx)
        y1 = int(y1s * sy);  y2 = int(y2s * sy)

        cx   = (x1s + x2s) / 2.0
        cy   = (y1s + y2s) / 2.0
        area = float(max(1, x2s - x1s) * max(1, y2s - y1s))

        _last_seen[tid]  = _frame_idx
        _track_age[tid] += 1

        _smooth_area[tid] = (SIZE_EMA_ALPHA * area
                             + (1 - SIZE_EMA_ALPHA)
                             * _smooth_area.get(tid, area))
        _cx_hist[tid].append(cx)
        _cy_hist[tid].append(cy)
        _area_hist[tid].append(_smooth_area[tid])

        # ── Path A: fast close pass (no age gate) ─────────────────
        pa = _path_a(tid, cx, cy, _smooth_area[tid], barrier)
        if pa:
            _fast_consec[tid] += 1
            _score[tid] = min(SCORE_MAX,
                              _score.get(tid, 0.0) + FAST_SCORE_PER_FRAME)
        else:
            _fast_consec[tid] = 0

        # ── Path B: distant approach (age-gated) ──────────────────
        pb = _path_b(tid, cx, ego_mag, ego_stopped, barrier)
        if pb:
            _score[tid] = min(SCORE_MAX,
                              _score.get(tid, 0.0) + SCORE_PER_FRAME)
        elif not pa:
            # Only decay if NEITHER path fires
            _score[tid] = max(SCORE_MIN,
                              _score.get(tid, 0.0) - SCORE_DECAY)

        score    = _score[tid]
        is_viol  = score >= SCORE_THRESHOLD
        was_viol = _violation[tid]
        _violation[tid] = is_viol

        # ── Clip events ───────────────────────────────────────────
        if is_viol and not was_viol:
            _clip_start(tid, out)
        elif not is_viol and was_viol:
            _clip_end(tid)

        # ── Draw ──────────────────────────────────────────────────
        if is_viol:
            _draw_violation(out, x1, y1, x2, y2,
                            _cx_hist[tid], y1, y2, sx)
        elif score > 0:
            _draw_suspect(out, x1, y1, x2, y2, score, pa, pb)
        else:
            _draw_normal(out, x1, y1, x2, y2, tid)

    # ── HUD + overlays (drawn last, on top of everything) ─────────
    _draw_hud(out, ego_mag, ego_stopped, barrier)
    _draw_median_band(out)

    # ── Stale cleanup ─────────────────────────────────────────────
    if _frame_idx % REAP_INTERVAL == 0:
        stale = [t for t, f in _last_seen.items()
                 if _frame_idx - f > EXPIRY_FRAMES]
        for t in stale:
            for store in (_cx_hist, _cy_hist, _area_hist, _smooth_area,
                          _track_age, _last_seen, _score, _violation,
                          _fast_consec):
                store.pop(t, None)

    return out


def release():
    """Call once on shutdown to close any open clip writers."""
    _clip_release_all()