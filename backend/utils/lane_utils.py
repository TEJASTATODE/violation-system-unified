import cv2
import numpy as np

# ── Zone boundaries ───────────────────────────────────
LEFT_BOUNDARY  = 0.45
RIGHT_BOUNDARY = 0.55

# ── Colors ────────────────────────────────────────────
LEFT_COLOR      = (255, 50,  0)
CENTER_COLOR    = (0,   200, 0)
RIGHT_COLOR     = (0,   50,  255)
DIVIDER_COLOR   = (0,   255, 255)
APPROACH_COLOR  = (0,   165, 255)
VIOLATION_COLOR = (0,   0,   255)
ZONE_ALPHA      = 0.12

# ── Thresholds ────────────────────────────────────────
APPROACH_RATIO  = 1.06
MIN_AREA        = 3000
MIN_DY          = 1
MIN_MOVEMENT    = 3.0    # min pixels/frame to be moving
MIN_DX          = 2.0    # min horizontal movement


# ── White line center history ─────────────────────────
center_history  = []
SMOOTH_FRAMES   = 20


def detect_white_line_center(frame):
    """
    Detects white center divider line.
    Returns x position of divider.
    Falls back to frame center if not found.

    Pipeline:
    1. Convert to HSV
    2. Mask white pixels
    3. Apply ROI
    4. Hough lines on white pixels
    5. Find most central vertical line
    """
    h, w  = frame.shape[:2]

    # HSV white mask
    hsv         = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0,   0,   180])
    upper_white = np.array([180, 30,  255])
    white_mask  = cv2.inRange(hsv, lower_white, upper_white)

    # ROI — bottom 60% of frame
    roi = np.zeros_like(white_mask)
    polygon = np.array([[
        (0,          h),
        (w,          h),
        (int(w*0.65), int(h*0.40)),
        (int(w*0.35), int(h*0.40))
    ]], dtype=np.int32)
    cv2.fillPoly(roi, polygon, 255)
    masked = cv2.bitwise_and(white_mask, roi)

    # Hough lines
    lines = cv2.HoughLinesP(
        masked,
        rho=1, theta=np.pi/180,
        threshold=30,
        minLineLength=50,
        maxLineGap=100
    )

    if lines is None:
        return w // 2   # fallback

    # Find line closest to frame center
    frame_center = w // 2
    best_x       = None
    best_dist    = float("inf")

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1:
            continue
        slope = (y2-y1)/(x2-x1)

        # Only near-vertical lines
        if abs(slope) < 0.3:
            continue

        mid_x = (x1+x2)//2
        dist  = abs(mid_x - frame_center)

        if dist < best_dist:
            best_dist = dist
            best_x    = mid_x

    return best_x if best_x else frame_center


def get_smooth_center(frame):
    """
    Smooths center x over SMOOTH_FRAMES.
    Prevents jitter from frame to frame.
    """
    global center_history

    raw = detect_white_line_center(frame)
    center_history.append(raw)

    if len(center_history) > SMOOTH_FRAMES:
        center_history.pop(0)

    return int(np.mean(center_history))


def get_zone_boundaries(frame_width, center_x=None):
    """
    Returns zone boundaries.
    Uses detected center if available.
    """
    if center_x is None:
        center_x = frame_width // 2

    # 10% buffer around center
    buffer  = int(frame_width * 0.05)
    left_x  = center_x - buffer
    right_x = center_x + buffer

    return left_x, right_x


def get_vehicle_zone(box, frame_width, center_x=None):
    """
    Returns zone: left / center / right
    Based on vehicle center vs road center.
    """
    cx             = (box[0]+box[2])//2
    left_x, right_x = get_zone_boundaries(
        frame_width, center_x
    )

    if cx < left_x:
        return "left"
    elif cx < right_x:
        return "center"
    else:
        return "right"


def get_box_area(box):
    x1, y1, x2, y2 = box
    return max(0, x2-x1) * max(0, y2-y1)


def get_box_center(box):
    x1, y1, x2, y2 = box
    return ((x1+x2)//2, (y1+y2)//2)


def is_approaching(curr_area, prev_area,
                   curr_cy,   prev_cy):
    if prev_area <= 0:
        return False
    area_ratio = curr_area / prev_area
    area_grows = area_ratio >= APPROACH_RATIO
    moves_down = (curr_cy - prev_cy) >= MIN_DY
    return area_grows and moves_down


def is_moving_away(curr_area, prev_area,
                   curr_cy,   prev_cy):
    if prev_area <= 0:
        return False
    area_ratio   = curr_area / prev_area
    area_shrinks = area_ratio <= 0.94
    moves_up     = (prev_cy - curr_cy) >= 1
    return area_shrinks and moves_up


def draw_zones(frame, center_x=None):
    """
    Draws 3 zones using detected center.
    """
    h, w             = frame.shape[:2]
    if center_x is None:
        center_x     = w // 2
    left_x, right_x  = get_zone_boundaries(w, center_x)
    overlay          = frame.copy()

    # Left zone
    cv2.rectangle(overlay, (0,0), (left_x,h),
                  LEFT_COLOR, -1)

    # Center zone
    cv2.rectangle(overlay, (left_x,0), (right_x,h),
                  CENTER_COLOR, -1)

    # Right zone
    cv2.rectangle(overlay, (right_x,0), (w,h),
                  RIGHT_COLOR, -1)

    cv2.addWeighted(overlay, ZONE_ALPHA,
                    frame, 1-ZONE_ALPHA, 0, frame)

    # Divider lines
    cv2.line(frame, (left_x,0),  (left_x,h),
             DIVIDER_COLOR, 2)
    cv2.line(frame, (right_x,0), (right_x,h),
             DIVIDER_COLOR, 2)

    # Center line
    cv2.line(frame, (center_x,0), (center_x,h),
             (255,255,255), 1)

    # Labels
    cv2.putText(frame, "ONCOMING",
                (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, LEFT_COLOR, 2)

    cv2.putText(frame, "CENTER",
                (left_x+5, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, CENTER_COLOR, 2)

    cv2.putText(frame, "YOUR LANE",
                (right_x+5, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, RIGHT_COLOR, 2)

    return frame


def draw_vehicle_box(frame, box, track_id,
                     zone, approaching,
                     violation=False):
    x1, y1, x2, y2 = box
    cx, cy          = get_box_center(box)

    if violation:
        color     = VIOLATION_COLOR
        label     = f"WRONG SIDE ID:{track_id}"
        thickness = 3
    elif approaching:
        color     = APPROACH_COLOR
        label     = f"APPROACHING ID:{track_id}"
        thickness = 2
    else:
        color     = (0, 255, 0)
        label     = f"ID:{track_id} {zone.upper()}"
        thickness = 1

    cv2.rectangle(frame, (x1,y1), (x2,y2),
                  color, thickness)

    (tw, th), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
    )
    cv2.rectangle(frame,
                  (x1, y1-th-8),
                  (x1+tw+6, y1),
                  color, -1)
    b  = 0.299*color[2] + 0.587*color[1] + 0.114*color[0]
    tc = (0,0,0) if b > 127 else (255,255,255)
    cv2.putText(frame, label,
                (x1+3, y1-4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45, tc, 1)

    if approaching or violation:
        cv2.arrowedLine(frame,
                        (cx, y1-25),
                        (cx, y1-5),
                        color, 2, tipLength=0.4)

    return frame


def draw_status_panel(frame, total,
                      approaching, violations):
    data = [
        (f"Vehicles   : {total}",      (200,200,200)),
        (f"Approaching: {approaching}", APPROACH_COLOR),
        (f"Violations : {violations}",  VIOLATION_COLOR),
    ]

    overlay = frame.copy()
    cv2.rectangle(overlay, (5,50), (230,140),
                  (15,15,15), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.rectangle(frame, (5,50), (230,140),
                  (80,80,80), 1)

    for i, (text, color) in enumerate(data):
        cv2.putText(frame, text,
                    (12, 72+i*22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, color, 1)


def draw_violation_alert(frame, count):
    if count == 0:
        return
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0,0), (w,h),
                  VIOLATION_COLOR, 3)