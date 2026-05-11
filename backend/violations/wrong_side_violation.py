"""
wrong_side_violation.py
═══════════════════════════════════════════════════════════════════════

KEY CHANGE: PROXIMITY GATE
───────────────────────────
All previous versions tried to detect wrong-way vehicles at any distance.
This caused correct vehicles far away to get flagged because at distance:
  • Small bbox jitter looks like growth
  • Lane position is ambiguous (everything near vanishing point is "centre")
  • Ego motion correction is less reliable

The fix: a vehicle must prove it is CLOSE AND DANGEROUS before any
violation logic even runs. Distant vehicles are completely ignored.

PROXIMITY GATE (must pass before violation logic)
──────────────────────────────────────────────────
Three signals, each scores points. Vehicle needs PROX_MIN_SCORE to proceed:

  P1 — SIZE      bbox area > PROX_MIN_AREA_FRAC of frame      → 2 pts
  P2 — POSITION  centroid in lower PROX_LOWER_FRAC of frame   → 1 pt (soft)
  P3 — CLOSING   area growing over last few frames             → 2 pts

Max score = 5. PROX_MIN_SCORE = 3 means:
  • Large + closing (no position)  → 4 pts → passes  (hill/angle road)
  • Large + low position           → 3 pts → passes
  • Small + low + closing          → 3 pts → passes (approaching from far)
  • Small + not closing            → 0-1 pts → BLOCKED (distant vehicle)
  • Medium + not growing + high    → 2 pts → BLOCKED

VIOLATION LOGIC (only runs if proximity gate passes)
─────────────────────────────────────────────────────
Same dual-signal system with all 6 previous bugs fixed:
  Signal A — size growing faster than ego motion explains
  Signal B — centroid in our lane danger zone
  + Hysteresis on score to prevent flickering
  + cy tracked for proper 2D displacement
  + ego passed as parameter (not stale global)
  + HUD drawn last (on top of zone overlay)
"""

import cv2
import numpy as np
from collections import defaultdict, deque
from detection.detect import detect_general_objects
from utils.lane_utils import inside_roi

# ══════════════════════════════════════════════════════════
#  ROAD PROFILE
# ══════════════════════════════════════════════════════════
ACTIVE_PROFILE = "balanced"   # "narrow" | "balanced" | "highway"

_PROFILES = {
    "narrow": dict(
        lane_danger_left       = 0.25,
        approach_ratio_moving  = 1.12,
        approach_ratio_stopped = 1.06,
        score_per_frame        = 12,
        score_decay            = 3,
        score_threshold        = 60,
        score_hysteresis       = 15,
    ),
    "balanced": dict(
        lane_danger_left       = 0.33,
        approach_ratio_moving  = 1.18,
        approach_ratio_stopped = 1.08,
        score_per_frame        = 10,
        score_decay            = 3,
        score_threshold        = 70,
        score_hysteresis       = 15,
    ),
    "highway": dict(
        lane_danger_left       = 0.45,
        approach_ratio_moving  = 1.25,
        approach_ratio_stopped = 1.12,
        score_per_frame        = 8,
        score_decay            = 3,
        score_threshold        = 75,
        score_hysteresis       = 15,
    ),
}

P = _PROFILES[ACTIVE_PROFILE]

# ══════════════════════════════════════════════════════════
#  PROXIMITY GATE CONFIG  ← tune these first
# ══════════════════════════════════════════════════════════

# P1: minimum bbox area as fraction of total frame area
# 0.08 = vehicle must fill at least 8% of frame
# Raise to 0.12 to be stricter, lower to 0.05 to catch smaller vehicles
PROX_MIN_AREA_FRAC = 0.12

# P2: centroid must be in lower this fraction of frame (soft signal)
# 0.60 means centroid y > 40% from top
# Kept as soft (1pt only) because camera angle varies
PROX_LOWER_FRAC    = 0.65

# P3: area must have grown by at least this ratio over last N frames
PROX_GROWTH_RATIO  = 1.1   # 5% growth = closing in
PROX_GROWTH_FRAMES = 5      # compare latest area vs N frames ago

# Minimum proximity score to proceed to violation logic (max possible = 5)
PROX_MIN_SCORE     = 3

# ══════════════════════════════════════════════════════════
#  GENERAL CONSTANTS
# ══════════════════════════════════════════════════════════

MIN_TRACK_AGE    = 6
HISTORY_LEN      = 20
MIN_HISTORY      = 6
MIN_DISPLACEMENT = 5.0
EGO_STOPPED_PX   = 2.0
FLOW_SCALE       = 0.5
EGO_EMA_ALPHA    = 0.25
SIZE_EMA_ALPHA   = 0.15
EXPIRY_FRAMES    = 60
REAP_INTERVAL    = 30
SCORE_MAX        = 100
SCORE_MIN        = 0
# ══════════════════════════════════════════════════════════
#  STATE
# ══════════════════════════════════════════════════════════
_cx_history   = defaultdict(lambda: deque(maxlen=HISTORY_LEN))
_cy_history   = defaultdict(lambda: deque(maxlen=HISTORY_LEN))
_area_history = defaultdict(lambda: deque(maxlen=HISTORY_LEN))
_smooth_area  = {}
_track_age    = defaultdict(int)
_last_seen    = {}
_score        = defaultdict(float)
_violation    = defaultdict(bool)
_correct      = defaultdict(bool)

_frame_idx    = 0
_prev_gray    = None
_ego_motion   = 0.0


# ══════════════════════════════════════════════════════════
#  EGO MOTION
# ══════════════════════════════════════════════════════════
def _estimate_ego(prev_gray, curr_gray):
    global _ego_motion
    scale = FLOW_SCALE
    pg = cv2.resize(prev_gray, None, fx=scale, fy=scale)
    cg = cv2.resize(curr_gray, None, fx=scale, fy=scale)

    p0 = cv2.goodFeaturesToTrack(pg, 200, 0.01, 5)
    if p0 is None or len(p0) < 10:
        return _ego_motion
    p1, st, _ = cv2.calcOpticalFlowPyrLK(pg, cg, p0, None)
    if p1 is None:
        return _ego_motion

    p0g, p1g = p0[st == 1], p1[st == 1]
    if len(p0g) < 6:
        return _ego_motion

    H, mask = cv2.findHomography(p0g, p1g, cv2.RANSAC, 3.0)
    if H is None or mask is None:
        dx = float(np.median(p1g[:, 0] - p0g[:, 0]))
        dy = float(np.median(p1g[:, 1] - p0g[:, 1]))
    else:
        inl = mask.ravel().astype(bool)
        if inl.sum() < 4:
            return _ego_motion
        dx = float(np.median(p1g[inl, 0] - p0g[inl, 0]))
        dy = float(np.median(p1g[inl, 1] - p0g[inl, 1]))

        raw = abs(dy)/scale
    _ego_motion = EGO_EMA_ALPHA * raw + (1 - EGO_EMA_ALPHA) * _ego_motion
    return _ego_motion


# ══════════════════════════════════════════════════════════
#  PROXIMITY GATE
# ══════════════════════════════════════════════════════════
def _proximity_score(tid, cx, cy, smooth_area, frame_w, frame_h):
    """
    Returns (prox_score, p1, p2, p3) where prox_score is 0-5.
    Vehicle must reach PROX_MIN_SCORE to enter violation logic.

    This is the primary defence against false flags on distant vehicles.
    """
    frame_area = frame_w * frame_h
    areas      = list(_area_history[tid])
    score      = 0

    # P1: size — is vehicle large enough to be close?
    p1 = smooth_area >= (PROX_MIN_AREA_FRAC * frame_area)
    if p1:
        score += 2

    # P2: vertical position — is vehicle in lower portion of frame?
    # Soft signal (1pt) because camera angle varies
    p2 = cy >= (frame_h * (1.0 - PROX_LOWER_FRAC))
    if p2:
        score += 1

    # P3: closing in — area growing over last N frames?
    p3 = False
    if len(areas) >= PROX_GROWTH_FRAMES:
        old_area = float(np.mean(
            areas[:max(1, len(areas) - PROX_GROWTH_FRAMES)]))
        new_area = float(np.mean(areas[-3:]))
        if old_area > 0:
            p3 = (new_area / old_area) >= PROX_GROWTH_RATIO
    if p3:
        score += 2

    return score, p1, p2, p3


# ══════════════════════════════════════════════════════════
#  VIOLATION SIGNALS
# ══════════════════════════════════════════════════════════

# Maximum lateral drift (px) over history window for a vehicle
# to still be considered "in our path" (not passing us laterally).
# Opposite-lane vehicles passing us move much more than this.
# Raise if wrong-way vehicles on wide roads are being missed.
LATERAL_DRIFT_MAX = 0.12  # fraction of frame width


def _check_signals(tid, cx, frame_w, ego_mag, ego_stopped):
    """
    Returns (signal_a, signal_b).

    THE KEY FIX — lateral motion gate (signal_c):
    ───────────────────────────────────────────────
    Without a centre line, we cannot use lane position alone.
    An opposite-lane vehicle going the RIGHT way for them will:
      • Appear on the right side of frame (narrow road)  → triggers signal_b
      • Grow as they approach                            → triggers signal_a
    This is IDENTICAL to a wrong-way vehicle's signature — hence false flags.

    The discriminator: LATERAL MOVEMENT
      Wrong-way vehicle in our lane:
        → Stays roughly centred in frame as it approaches
        → Lateral drift is SMALL (it's coming straight at us)

      Correct opposite-lane vehicle passing us:
        → Moves sideways across frame as it passes
        → Lateral drift is LARGE (it slides past our side)

    We require lateral drift to be SMALL before flagging.
    If a vehicle is drifting sideways significantly, it is
    passing us in its own lane — not threatening our path.
    """
    if len(_area_history[tid]) < MIN_HISTORY:
        return False, False, False

    # Signal B: in our lane zone (right of danger boundary)
    lane_boundary = P["lane_danger_left"] * frame_w
    cx_list = list(_cx_history[tid])
    if len(cx_list) >= MIN_HISTORY:
     avg_cx = sum(cx_list[-5:]) / 5
     signal_b = avg_cx >= lane_boundary
     
    else:
       signal_b = False


    # Signal C (NEW): lateral stability — vehicle stays in our path
    # Measure total horizontal drift over the history window.
    cx_list = list(_cx_history[tid])
    if len(cx_list) >= MIN_HISTORY:
        lateral_total = abs(cx_list[-1] - cx_list[0])
        lateral_frac  = lateral_total / max(frame_w, 1)
        # If the vehicle has moved more than LATERAL_DRIFT_MAX
        # across the frame, it is passing us — not coming at us.
        signal_c = lateral_frac <= LATERAL_DRIFT_MAX
    else:
        signal_c = True   # not enough history yet — don't block

    if not signal_c:
        return False, signal_b, False   # passing vehicle — not wrong-way

    # Signal A: approaching (size growing faster than ego explains)
    areas    = list(_area_history[tid])
    area_old = float(np.mean(areas[:3]))
    area_new = float(np.mean(areas[-3:]))
    if area_old <= 0:
        return False, signal_b, signal_c

    growth = area_new / area_old

    if ego_stopped:
        threshold = P["approach_ratio_stopped"]
    else:
        ego_ratio     = ego_mag / max(frame_w, 1)
        ego_inflation = 1.0 + ego_ratio * 8.0
        threshold     = P["approach_ratio_moving"] * ego_inflation

    signal_a = growth >= threshold
    return signal_a, signal_b, signal_c


# ══════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════
def process_frame(frame, prev_frame):
    global _frame_idx, _prev_gray

    _frame_idx += 1
    h, w = frame.shape[:2]
    frame_area = w * h

    curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_gray = (_prev_gray if _prev_gray is not None
                 else cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY))
    _prev_gray = curr_gray

    ego_mag     = _estimate_ego(prev_gray, curr_gray)
    ego_stopped = ego_mag < EGO_STOPPED_PX

    detections = detect_general_objects(frame)
    out = frame.copy()

    for det in detections:
        tid = det.get("track_id", -1)
        if tid == -1:
            continue

        x1, y1, x2, y2 = det["box"]
        cx   = (x1 + x2) / 2.0
        cy   = (y1 + y2) / 2.0
        area = float(max(1, x2 - x1) * max(1, y2 - y1))

        _last_seen[tid]  = _frame_idx
        _track_age[tid] += 1

        _smooth_area[tid] = (
            SIZE_EMA_ALPHA * area
            + (1 - SIZE_EMA_ALPHA) * _smooth_area.get(tid, area)
        )
        _cx_history[tid].append(cx)
        _cy_history[tid].append(cy)
        _area_history[tid].append(_smooth_area[tid])


        # ── ADAPTIVE ROI GATE ──────────────────────────────────────
        # Reject vehicles above the estimated horizon entirely.
        # Vehicles above the horizon are too far or in sky — skip.
        if not inside_roi(cx, cy, frame):
            _draw_normal(out, x1, y1, x2, y2, tid)
            continue

        # Immature track — draw plain, skip all logic
        if _track_age[tid] < MIN_TRACK_AGE:
            _draw_normal(out, x1, y1, x2, y2, tid)
            continue

        # 2D displacement gate
        if len(_cx_history[tid]) >= 2:
            cxl = list(_cx_history[tid])
            cyl = list(_cy_history[tid])
            net = ((cxl[-1]-cxl[0])**2 + (cyl[-1]-cyl[0])**2) ** 0.5
            if net < MIN_DISPLACEMENT:
                _draw_normal(out, x1, y1, x2, y2, tid)
                continue

        # ── PROXIMITY GATE ─────────────────────────────────────────
        prox, p1, p2, p3 = _proximity_score(
            tid, cx, cy, _smooth_area[tid], w, h)

        if prox < PROX_MIN_SCORE:
            # Vehicle is too far / too small — draw grey, skip violation
            _draw_distant(out, x1, y1, x2, y2, prox)
            # Also reset score so a vehicle can't accumulate while distant
            _score[tid]     = max(SCORE_MIN, _score.get(tid, 0.0) - P["score_decay"])
            _violation[tid] = False
            continue

        # ── BIDIRECTIONAL CLASSIFICATION ───────────────────────────
        # We classify every vehicle into one of three states:
        #   CORRECT  — confirmed moving same direction as ego (green)
        #   WRONG    — confirmed moving against traffic (red)
        #   UNKNOWN  — not enough evidence yet (no box drawn)
        #
        # This is better than "not wrong = correct" because it requires
        # positive evidence for BOTH classifications. A vehicle that is
        # ambiguous (turning, stopped, or new) shows nothing — no clutter.

        areas = list(_area_history[tid])

        # ── CORRECT vehicle signals ────────────────────────────────
        # C1: bbox SHRINKING — moving away from dashcam
        # C3: HIGH lateral drift — passing us in their own lane
        #     This directly catches the opposite-lane case:
        #     they slide across frame quickly as we pass each other.

        c1 = False
        if len(areas) >= MIN_HISTORY:
            old_a = float(np.mean(areas[:3]))
            new_a = float(np.mean(areas[-3:]))
            if old_a > 0:
                c1 = (old_a / max(new_a, 1)) >= 1.05   # shrinking

        # C3: significant lateral movement = passing vehicle (correct)
        cx_list  = list(_cx_history[tid])
        c3 = False
        if len(cx_list) >= MIN_HISTORY:
            lateral_frac = abs(cx_list[-1] - cx_list[0]) / max(w, 1)
            c3 = lateral_frac > LATERAL_DRIFT_MAX   # mirror of signal_c

        correct_score_delta = 0
        if c1:
            correct_score_delta += P["score_per_frame"]
        if c3:   # lateral pass = strong correct signal regardless of shrink
            correct_score_delta += P["score_per_frame"]
        if c1 and c3:   # both = very confident
            correct_score_delta += P["score_per_frame"] // 2

        # ── WRONG vehicle signals ──────────────────────────────────
        sig_a, sig_b, sig_c = _check_signals(tid, cx, w, ego_mag, ego_stopped)
        both_active  = sig_a and sig_b and sig_c

        # ── SCORE UPDATE ───────────────────────────────────────────
        # Positive score  → wrong-way evidence
        # Negative score  → correct-vehicle evidence
        # Near zero       → unknown / ambiguous
        if both_active:
            _score[tid] = min(SCORE_MAX,
                              _score.get(tid, 0.0) + P["score_per_frame"])
        elif correct_score_delta > 0:
            # Correct signals push score toward negative
            _score[tid] = max(-SCORE_MAX,
                              _score.get(tid, 0.0) - correct_score_delta)
        else:
            # No strong signal either way — decay toward zero
            _score[tid] = _score.get(tid, 0.0) * 0.85

        score = _score[tid]

        # ── CLASSIFY ───────────────────────────────────────────────
        # Hysteresis bands prevent flickering at thresholds
        was_viol    = _violation.get(tid, False)
        was_correct = _correct.get(tid, False)

        if was_viol:
            violation = score >= (P["score_threshold"] - P["score_hysteresis"])
        else:
            violation = score >= P["score_threshold"]

        if was_correct:
            confirmed_correct = score <= -(P["score_threshold"] - P["score_hysteresis"])
        else:
            confirmed_correct = score <= -P["score_threshold"]

        _violation[tid] = violation
        _correct[tid]   = confirmed_correct

        # ── DRAW ───────────────────────────────────────────────────
        if violation:
            _draw_active(out, x1, y1, x2, y2, tid, score,
                         True, sig_a, sig_b, sig_c, _cx_history[tid])
        elif confirmed_correct:
            _draw_correct(out, x1, y1, x2, y2, tid)
        # else: ambiguous — draw nothing (cleanest output)

    # Draw zone overlay first, then HUD on top
    _draw_zone(out, w, h)
    _draw_hud(out, ego_mag, ego_stopped)

    # Cleanup
    if _frame_idx % REAP_INTERVAL == 0:
        stale = [t for t, f in _last_seen.items()
                 if _frame_idx - f > EXPIRY_FRAMES]
        for t in stale:
            for store in (_cx_history, _cy_history, _area_history,
                          _smooth_area, _track_age, _last_seen,
                          _score, _violation, _correct):
                store.pop(t, None)

    return out


# ══════════════════════════════════════════════════════════
#  DRAWING HELPERS
# ══════════════════════════════════════════════════════════

def _corner_box(frame, x1, y1, x2, y2, color, thickness=1, corner_len=10):
    """
    Draws only the four corner brackets of a bounding box.
    Much cleaner than a full rectangle — less visual noise,
    easier to see the actual vehicle underneath.
    corner_len: length of each corner arm in pixels.
    """
    tl = (x1, y1); tr = (x2, y1)
    bl = (x1, y2); br = (x2, y2)
    cl = corner_len
    t  = thickness

    # Top-left
    cv2.line(frame, tl, (x1 + cl, y1), color, t)
    cv2.line(frame, tl, (x1, y1 + cl), color, t)
    # Top-right
    cv2.line(frame, tr, (x2 - cl, y1), color, t)
    cv2.line(frame, tr, (x2, y1 + cl), color, t)
    # Bottom-left
    cv2.line(frame, bl, (x1 + cl, y2), color, t)
    cv2.line(frame, bl, (x1, y2 - cl), color, t)
    # Bottom-right
    cv2.line(frame, br, (x2 - cl, y2), color, t)
    cv2.line(frame, br, (x2, y2 - cl), color, t)


def _label_pill(frame, text, x, y, bg_color, text_color=(255, 255, 255),
                font_scale=0.42, thickness=1):
    """
    Draws a filled pill-shaped background behind text.
    Looks cleaner than raw putText on video frames.
    Returns the width of the drawn pill.
    """
    font  = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    pad_x, pad_y = 5, 3
    rx1 = x
    ry1 = y - th - pad_y
    rx2 = x + tw + pad_x * 2
    ry2 = y + pad_y
    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), bg_color, -1)
    cv2.putText(frame, text, (x + pad_x, y),
                font, font_scale, text_color, thickness, cv2.LINE_AA)
    return rx2 - rx1


# ── State colours (BGR) ───────────────────────────────────
_COL_NORMAL  = (80, 200, 80)      # safe — thin green
_COL_DISTANT = (80, 80, 80)       # grey — too far, ignored
_COL_WATCH   = (0, 180, 255)      # amber/orange — building score
_COL_WRONG   = (30, 30, 255)      # red — confirmed violation


def _draw_normal(frame, x1, y1, x2, y2, tid):
    """
    Immature / far-below-proximity track.
    Corner brackets only, 1px, muted green. No label.
    """
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    _corner_box(frame, x1, y1, x2, y2, _COL_NORMAL,
                thickness=1, corner_len=8)


def _draw_correct(frame, x1, y1, x2, y2, tid):
    """
    Confirmed correct vehicle — moving same direction as ego.
    Clean green corner brackets + small 'OK' pill.
    Deliberately subtle — correct vehicles should not grab attention.
    """
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    bw = x2 - x1
    cl = max(8, min(16, bw // 5))
    _corner_box(frame, x1, y1, x2, y2, _COL_NORMAL,
                thickness=1, corner_len=cl)
    # Tiny 'OK' label — readable but unobtrusive
    _label_pill(frame, "OK", x1, max(y1 - 4, 14),
                bg_color=(40, 140, 40),
                text_color=(255, 255, 255),
                font_scale=0.34, thickness=1)


def _draw_distant(frame, x1, y1, x2, y2, prox_score):
    """
    Detected but too small/far to analyse.
    Tiny grey corner brackets — barely visible, not distracting.
    """
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    _corner_box(frame, x1, y1, x2, y2, _COL_DISTANT,
                thickness=1, corner_len=6)


def _draw_active(frame, x1, y1, x2, y2, tid, score,
                 violation, sig_a, sig_b, sig_c, cx_hist):
    """
    Vehicle passed proximity gate — render full detection UI.

    Normal (score building):
      Amber corner brackets + small score pill above box.

    Violation confirmed:
      Bold red corner brackets + filled 'WRONG WAY' pill
      + dotted centroid trail showing direction of travel.
    """
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    bw = x2 - x1
    bh = y2 - y1
    # Corner length scales with box size — larger vehicle = bigger corners
    cl = max(8, min(18, bw // 5))

    if violation:
        color = _COL_WRONG

        # Corner brackets — 2px bold red
        _corner_box(frame, x1, y1, x2, y2, color, thickness=2, corner_len=cl)

        # 'WRONG WAY' pill label above box
        label = "WRONG WAY"
        pill_y = max(y1 - 4, 22)
        _label_pill(frame, label, x1, pill_y,
                    bg_color=color, text_color=(255, 255, 255),
                    font_scale=0.50, thickness=1)

        # Centroid trail — dashed line showing travel path
        pts    = list(cx_hist)
        cy_mid = y1 + bh // 2
        for i in range(1, len(pts)):
            # Every other segment drawn = dashed effect
            if i % 2 == 0:
                continue
            cv2.line(frame,
                     (int(pts[i - 1]), cy_mid),
                     (int(pts[i]),     cy_mid),
                     (100, 100, 255), 1, cv2.LINE_AA)

    else:
        # Score-based colour: green → amber → orange as score builds
        ratio = min(score / max(P["score_threshold"], 1), 1.0)
        r     = int(30  + 220 * ratio)
        g     = int(200 - 160 * ratio)
        color = (0, g, r)   # BGR

        # Corner brackets — 1px
        _corner_box(frame, x1, y1, x2, y2, color, thickness=1, corner_len=cl)

        # Score pill — only show if score > 10 (ignore near-zero noise)
        if score > 10:
            pill_txt = f"S {int(score)}"
            pill_y   = max(y1 - 4, 16)
            _label_pill(frame, pill_txt, x1, pill_y,
                        bg_color=color, text_color=(255, 255, 255),
                        font_scale=0.36, thickness=1)


def _draw_zone(frame, w, h):
    """Subtle lane danger-zone overlay — drawn before HUD."""
    danger_x = int(P["lane_danger_left"] * w)
    overlay  = frame.copy()
    cv2.rectangle(overlay, (danger_x, 0), (w, h), (20, 20, 180), -1)
    cv2.addWeighted(overlay, 0.04, frame, 0.96, 0, frame)
    # Single pixel vertical divider
    cv2.line(frame, (danger_x, 0), (danger_x, h), (60, 60, 200), 1)


def _draw_hud(frame, ego_mag, ego_stopped):
    """
    Clean HUD panel — semi-transparent background pill,
    drawn last so it always sits on top.
    """
    h, w = frame.shape[:2]

    status  = "STOPPED" if ego_stopped else f"MOVING  {ego_mag:.1f} px/f"
    lines   = [
        f"Ego  {status}",
        f"Profile  {ACTIVE_PROFILE}   ProxThr  {PROX_MIN_SCORE}/5",
    ]
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.40
    pad_x, pad_y = 8, 6
    line_h     = 16

    # Panel background
    panel_w = 300
    panel_h = pad_y * 2 + line_h * len(lines)
    overlay  = frame.copy()
    cv2.rectangle(overlay, (6, 6), (6 + panel_w, 6 + panel_h),
                  (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    for i, txt in enumerate(lines):
        cv2.putText(frame, txt,
                    (6 + pad_x, 6 + pad_y + line_h * (i + 1) - 2),
                    font, font_scale, (200, 220, 220), 1, cv2.LINE_AA)