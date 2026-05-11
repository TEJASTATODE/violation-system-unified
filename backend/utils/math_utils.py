def boxes_overlap(box1, box2):
    """Simple overlap check — do boxes touch at all?"""
    x1, y1, x2, y2 = box1
    a1, b1, a2, b2 = box2
    return x1 < a2 and x2 > a1 and y1 < b2 and y2 > b1


def box_inside(inner_box, outer_box):
    """Is inner box completely inside outer box?"""
    x1, y1, x2, y2 = inner_box
    a1, b1, a2, b2 = outer_box
    return x1 >= a1 and y1 >= b1 and x2 <= a2 and y2 <= b2


def get_head_box(person_box, ratio=None, frame_w=1920, frame_h=1080):
    """
    Adaptive head box for dashcam front view.
    Dashcam sees front of person — more head/face visible
    so ratios are slightly larger than CCTV overhead view.

    Small/distant  (height < 80)  → ratio 0.45, big padding
    Medium         (height < 150) → ratio 0.38, normal padding
    Close/large    (height >= 150) → ratio 0.32, small padding

    Clamps all edges to frame boundaries so downstream model
    never receives out-of-bounds coordinates.
    """
    x1, y1, x2, y2 = person_box
    height = y2 - y1
    width  = x2 - x1

    if ratio is None:
        if height < 80:
            ratio = 0.45
        elif height < 150:
            ratio = 0.38
        else:
            ratio = 0.32

    if height < 80:
        pad = int(width * 0.20)
    elif height < 150:
        pad = int(width * 0.10)
    else:
        pad = int(width * 0.08)

    head_y2 = int(y1 + ratio * height)

    # Clamp all edges to frame dimensions — prevents silent out-of-bounds
    return (
        max(0,       x1 - pad),
        max(0,       y1),
        min(frame_w, x2 + pad),   # FIX: was unbounded on the right
        min(frame_h, head_y2),
    )


def compute_iou(box1, box2):
    """
    Intersection over Union — 0.0 to 1.0.
    """
    x1, y1, x2, y2 = box1
    a1, b1, a2, b2 = box2

    ix1 = max(x1, a1)
    iy1 = max(y1, b1)
    ix2 = min(x2, a2)
    iy2 = min(y2, b2)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0

    area1 = (x2 - x1) * (y2 - y1)
    area2 = (a2 - a1) * (b2 - b1)
    union = area1 + area2 - inter

    return inter / union


def compute_overlap_ratio(box1, box2):
    """
    What fraction of box1 is covered by box2?
    Useful when box1 (e.g. head) is partially inside box2 (e.g. helmet).
    """
    x1, y1, x2, y2 = box1
    a1, b1, a2, b2 = box2

    ix1 = max(x1, a1)
    iy1 = max(y1, b1)
    ix2 = min(x2, a2)
    iy2 = min(y2, b2)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0

    area1 = (x2 - x1) * (y2 - y1)
    if area1 == 0:
        return 0.0

    return inter / area1


def get_box_center(box):
    """Returns center point of a box."""
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def get_box_area(box):
    """Returns area of a box."""
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)