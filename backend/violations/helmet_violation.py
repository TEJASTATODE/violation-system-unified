from utils.math_utils import (boxes_overlap, get_head_box,
                               compute_iou, compute_overlap_ratio)
from collections import defaultdict

GENERAL_CONF    = 0.30   # stricter for dashcam
HELMET_CONF     = 0.35
MIN_SCORE       = 0.02
CONFIRM_FRAMES  = 2
VIOLATION_RATIO = 1.5

# Per rider tracking
locked_riders = {}
pending       = defaultdict(list)


def get_rider_key(person_box, track_id=-1, grid=80):
    """
    Use track_id if available — stable for dashcam movement
    Fallback to center grid if no track_id
    """
    if track_id != -1:
        return f"track_{track_id}"  

    # Fallback — center based
    cx = (person_box[0] + person_box[2]) // 2
    cy = (person_box[1] + person_box[3]) // 2
    return ((cx//grid)*grid, (cy//grid)*grid)


def get_best_match(head_box, helmet_boxes, person_box):
    if not helmet_boxes:
        return 0.0

    best   = 0.0
    x1, y1, x2, y2 = head_box
    head_h = y2 - y1
    head_w = x2 - x1

    px1, py1, px2, py2 = person_box
    pw = px2 - px1
    ph = py2 - py1

    person_region = (
        max(0, px1 - int(pw*0.15)),
        max(0, py1 - int(ph*0.05)),
        px2 + int(pw*0.15),
        py2 + int(ph*0.05)
    )

    pad = int(head_w*0.25) if head_h < 30 else int(head_w*0.08)
    expanded = (
        max(0, x1-pad),
        max(0, y1-pad),
        x2+pad,
        y2+pad
    )

    for hbox in helmet_boxes:
        if not boxes_overlap(hbox, person_region):
            continue

        iou         = compute_iou(head_box, hbox)
        overlap     = compute_overlap_ratio(head_box, hbox)
        exp_iou     = compute_iou(expanded, hbox)
        exp_overlap = compute_overlap_ratio(expanded, hbox)
        score       = max(iou, overlap, exp_iou*0.8, exp_overlap*0.7)

        if score > best:
            best = score

    return best


def is_rider(person_box, bike_box):
    px1, py1, px2, py2 = person_box
    ph       = py2 - py1
    p_cutoff = int(py1 + ph * 0.50)
    p_bottom = (px1, p_cutoff, px2, py2)
    ratio    = compute_overlap_ratio(p_bottom, bike_box)
    simple   = boxes_overlap(person_box, bike_box)
    return ratio >= 0.20 or simple


def make_decision(with_score, without_score):
    if with_score == 0.0 and without_score == 0.0:
        return None, 0.0

    if with_score >= MIN_SCORE:
        if without_score > with_score * VIOLATION_RATIO:
            return "violation", without_score
        else:
            return "safe", with_score

    if without_score >= MIN_SCORE:
        return "violation", without_score

    return None, 0.0


def helmet_violation(general_detections, helmet_detections):
    global locked_riders, pending

    persons        = []
    motorcycles    = []
    with_helmet    = []
    without_helmet = []

    for det in general_detections:
        if det["conf"] < GENERAL_CONF:
            continue
        if det["class"] == 0:
            # ── Keep full det with track_id ───────────────
            persons.append(det)
        elif det["class"] == 3:
            motorcycles.append(det["box"])

    for det in helmet_detections:
        if det["conf"] < HELMET_CONF:
            continue
        if det["class"] == 0:
            with_helmet.append(det["box"])
        elif det["class"] == 1:
            without_helmet.append(det["box"])

    # Find riders
    riders = []
    for person in persons:
        for bike in motorcycles:
            if is_rider(person["box"], bike):
                riders.append(person)
                break

    violations  = []
    safe_riders = []

    for rider in riders:
        # ── Use track_id for stable key ───────────────────
        key      = get_rider_key(
                       rider["box"],
                       track_id = rider.get("track_id", -1)
                   )
        head_box = get_head_box(rider["box"])

        # Already locked → use saved decision
        if key in locked_riders:
            d = locked_riders[key]
            if d["decision"] == "violation":
                violations.append({
                    "rider_box": rider["box"],
                    "head_box":  head_box,
                    "conf":      d["conf"]
                })
            else:
                safe_riders.append({
                    "rider_box": rider["box"],
                    "head_box":  head_box,
                    "conf":      d["conf"]
                })
            continue

        # Detect
        with_score    = get_best_match(head_box, with_helmet,    rider["box"])
        without_score = get_best_match(head_box, without_helmet, rider["box"])

        decision, conf = make_decision(with_score, without_score)

        if decision is None:
            continue

        # Confirm before locking
        pending[key].append(decision)

        if len(pending[key]) > CONFIRM_FRAMES:
            pending[key].pop(0)

        if (len(pending[key]) >= CONFIRM_FRAMES and
                len(set(pending[key])) == 1):
            locked_riders[key] = {
                "decision": decision,
                "conf":     conf
            }
            del pending[key]
            print(f"🔒 Locked {key} → {decision} ({conf:.0%})")

        # Show current decision while pending
        if decision == "violation":
            violations.append({
                "rider_box": rider["box"],
                "head_box":  head_box,
                "conf":      conf
            })
        else:
            safe_riders.append({
                "rider_box": rider["box"],
                "head_box":  head_box,
                "conf":      conf
            })

    return violations, safe_riders


def reset_locked_riders():
    global locked_riders, pending
    locked_riders = {}
    pending.clear()
    print("🔄 Reset all decisions!")