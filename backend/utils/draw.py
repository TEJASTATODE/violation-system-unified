import cv2
import numpy as np


def blend_rect(frame, x1, y1, x2, y2, color, alpha):
    """
    Alpha-blends a filled rectangle into `frame` in place, touching only the
    pixels inside (x1,y1)-(x2,y2).

    The common pattern elsewhere was `overlay = frame.copy(); cv2.rectangle(
    overlay, ...); cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)`
    — that copies and blends the ENTIRE frame just to tint a small HUD panel
    or zone. Outside the rectangle the math is a no-op (alpha*px + (1-alpha)*px
    == px), so it's wasted work — real cost on a CPU-only real-time pipeline
    already running 5 YOLO models per frame. This produces the identical
    output, only over the actual rectangle's pixels.
    """
    # cv2.rectangle treats (x1,y1)-(x2,y2) as inclusive on both corners;
    # match that exactly (+1 on the slice end) so this is a true drop-in
    # replacement regardless of whether a caller's x2/y2 is an interior
    # coordinate or the frame edge — an excl-end slice here silently left a
    # 1px unblended strip along the right/bottom edge of every panel that
    # didn't happen to end exactly at the frame boundary.
    h, w = frame.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w - 1, int(x2)), min(h - 1, int(y2))
    if x2 < x1 or y2 < y1:
        return
    roi = frame[y1:y2 + 1, x1:x2 + 1]
    overlay = np.full_like(roi, color)
    cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, roi)


def draw_box(frame, box, color=(0, 255, 0), label=None, thickness=1):
    """
    Thin precise bounding box with corner accents only.
    No filled background — clean minimal look.
    """
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1

    cl = max(8, int(min(w, h) * 0.15))

    # ── Draw thin main box ────────────────────────────
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # ── Corner accents (slightly thicker) ────────────
    ct = thickness + 1

    # Top left
    cv2.line(frame, (x1, y1),    (x1+cl, y1),  color, ct)
    cv2.line(frame, (x1, y1),    (x1, y1+cl),  color, ct)

    # Top right
    cv2.line(frame, (x2, y1),    (x2-cl, y1),  color, ct)
    cv2.line(frame, (x2, y1),    (x2, y1+cl),  color, ct)

    # Bottom left
    cv2.line(frame, (x1, y2),    (x1+cl, y2),  color, ct)
    cv2.line(frame, (x1, y2),    (x1, y2-cl),  color, ct)

    # Bottom right
    cv2.line(frame, (x2, y2),    (x2-cl, y2),  color, ct)
    cv2.line(frame, (x2, y2),    (x2, y2-cl),  color, ct)

    # ── Label ─────────────────────────────────────────
    if label:
        draw_label(frame, label, (x1, y1), color)


def draw_label(frame, text, pos, color=(0, 255, 0)):
    """
    Clean thin label with small background.
    """
    x, y = pos

    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4    # small text
    thickness  = 1      # thin text

    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)

    pad = 3

    
    lx1 = x
    ly1 = y - th - pad * 2
    lx2 = x + tw + pad * 2
    ly2 = y

    
    if ly1 < 0:
        ly1 = y + 1
        ly2 = y + th + pad * 2

    
    cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), color, -1)

    
    b = 0.299*color[2] + 0.587*color[1] + 0.114*color[0]
    tc = (0,0,0) if b > 127 else (255,255,255)

    cv2.putText(
        frame, text,
        (lx1+pad, ly2-pad),
        font, font_scale, tc, thickness
    )


def draw_text(frame, text, pos, color=(255, 255, 255)):
    """
    Clean thin status text — no background.
    """
    cv2.putText(
        frame, text, pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45, color, 1
    )


def draw_status_panel(frame, persons, bikes, helmets, violations, safe):
    """
    Minimal clean status panel — top left corner.
    """
    panel_x = 8
    panel_y = 8
    line_h  = 20
    pad     = 6
    width   = 180
    lines   = 5
    height  = pad*2 + line_h*lines

    # Semi transparent dark background
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (panel_x, panel_y),
        (panel_x+width, panel_y+height),
        (15, 15, 15), -1
    )
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Thin border
    cv2.rectangle(
        frame,
        (panel_x, panel_y),
        (panel_x+width, panel_y+height),
        (80, 80, 80), 1
    )

    # Status lines
    data = [
        (f"Persons   : {persons}",    (200, 200, 200)),
        (f"Bikes     : {bikes}",      (255, 165,   0)),
        (f"Helmets   : {helmets}",    (  0, 255,   0)),
        (f"Safe      : {safe}",       (  0, 255, 255)),
        (f"Violations: {violations}", (  0,   0, 255)),
    ]

    for i, (text, color) in enumerate(data):
        y = panel_y + pad + line_h * i + 14
        cv2.putText(
            frame, text,
            (panel_x+pad, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4, color, 1
        )


def draw_violation_alert(frame, count):
    """
    Thin red border alert when violations detected.
    """
    if count == 0:
        return

    h, w = frame.shape[:2]

    # Thin red border
    cv2.rectangle(frame, (0,0), (w,h), (0,0,255), 2)