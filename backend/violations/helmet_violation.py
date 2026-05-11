from utils.math_utils import (boxes_overlap, get_head_box,
                               compute_iou, compute_overlap_ratio)
from collections import defaultdict
from time import time

# ── Confidence thresholds ─────────────────────────────────────────────────────
GENERAL_CONF    = 0.30
HELMET_CONF     = 0.35
MIN_SCORE       = 0.02

# ── Minimum rider size ────────────────────────────────────────────────────────
MIN_RIDER_HEIGHT = 40
MIN_RIDER_WIDTH  = 15

# ── Confirmation logic ────────────────────────────────────────────────────────
CONFIRM_FRAMES   = 3
CONFIRM_MAJORITY = 2

# ── Decision logic ────────────────────────────────────────────────────────────
HELMET_DOMINANCE_THRESHOLD = 1.5

# ── TTL ───────────────────────────────────────────────────────────────────────
LOCK_TTL = 5.0

# ── Size bands ────────────────────────────────────────────────────────────────
CLOSE_RIDER_H  = 180
MEDIUM_RIDER_H = 80


class HelmetViolationDetector:

    def __init__(self,
                 frame_w: int = 1920,
                 frame_h: int = 1080,
                 lock_ttl: float = LOCK_TTL,
                 confirm_frames: int = CONFIRM_FRAMES,
                 confirm_majority: int = CONFIRM_MAJORITY,
                 min_rider_h: int = MIN_RIDER_HEIGHT,
                 min_rider_w: int = MIN_RIDER_WIDTH):

        self.frame_w          = frame_w
        self.frame_h          = frame_h
        self.lock_ttl         = lock_ttl
        self.confirm_frames   = confirm_frames
        self.confirm_majority = confirm_majority
        self.min_rider_h      = min_rider_h
        self.min_rider_w      = min_rider_w

        self._locked_riders: dict = {}
        self._pending: dict       = defaultdict(list)
        self._prev_size: dict     = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def process_frame(self,
                      general_detections: list,
                      helmet_detections:  list,
                      current_time:       float | None = None
                      ) -> tuple[list, list]:
        if current_time is None:
            current_time = time()

        self._expire_stale_locks(current_time)

        persons, motorcycles, with_helmet, without_helmet = \
            self._parse_detections(general_detections, helmet_detections)

        riders = self._find_riders(persons, motorcycles)

        # ── KEY FIX: assign each helmet box to its nearest rider first ────────
        # With 3-5 riders and full-body helmet boxes, checking all helmet boxes
        # against each rider causes cross-rider contamination. Instead we build
        # a 1-to-1 map: each helmet box goes to the single closest rider only.
        with_helmet_per_rider    = _assign_boxes_to_riders(with_helmet,    riders)
        without_helmet_per_rider = _assign_boxes_to_riders(without_helmet, riders)

        violations  = []
        safe_riders = []

        for i, rider in enumerate(riders):
            rx1, ry1, rx2, ry2 = rider["box"]
            rider_h = ry2 - ry1
            rider_w = rx2 - rx1

            if rider_h < self.min_rider_h or rider_w < self.min_rider_w:
                continue

            key      = self._get_rider_key(rider["box"],
                                           rider.get("track_id", -1))
            head_box = get_head_box(rider["box"],
                                    frame_w=self.frame_w,
                                    frame_h=self.frame_h)

            prev_h     = self._prev_size.get(key, rider_h)
            is_growing = rider_h > prev_h * 1.05
            self._prev_size[key] = rider_h

            # Already locked → refresh TTL and use saved decision
            if key in self._locked_riders:
                self._locked_riders[key]["ts"] = current_time
                d = self._locked_riders[key]
                entry = {"rider_box": rider["box"],
                         "head_box":  head_box,
                         "conf":      d["conf"]}
                (violations if d["decision"] == "violation"
                 else safe_riders).append(entry)
                continue

            # Only use helmet boxes assigned to THIS rider
            my_with    = with_helmet_per_rider.get(i, [])
            my_without = without_helmet_per_rider.get(i, [])

            with_score    = _get_best_match(head_box, my_with,    rider["box"],
                                            rider_h=rider_h, is_growing=is_growing)
            without_score = _get_best_match(head_box, my_without, rider["box"],
                                            rider_h=rider_h, is_growing=is_growing)

            # Scale threshold by size — close/growing riders need stronger signal
            if rider_h >= CLOSE_RIDER_H:
                threshold = HELMET_DOMINANCE_THRESHOLD * 1.6
            elif rider_h >= MEDIUM_RIDER_H:
                threshold = HELMET_DOMINANCE_THRESHOLD * 1.2
            else:
                threshold = HELMET_DOMINANCE_THRESHOLD
            if is_growing:
                threshold *= 1.2

            decision, conf = _make_decision(with_score, without_score, threshold)

            if decision is not None:
                buf = self._pending[key]
                buf.append(decision)
                if len(buf) > self.confirm_frames:
                    buf.pop(0)

                if len(buf) >= self.confirm_frames:
                    v_count = buf.count("violation")
                    s_count = buf.count("safe")
                    if v_count >= self.confirm_majority:
                        self._lock(key, "violation", conf, current_time)
                    elif s_count >= self.confirm_majority:
                        self._lock(key, "safe", conf, current_time)

                entry = {"rider_box": rider["box"],
                         "head_box":  head_box,
                         "conf":      conf}
                (violations if decision == "violation" else safe_riders).append(entry)

        return violations, safe_riders

    def reset(self) -> None:
        self._locked_riders.clear()
        self._pending.clear()
        self._prev_size.clear()
        print("Reset all decisions.")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _expire_stale_locks(self, current_time: float) -> None:
        stale = [k for k, v in self._locked_riders.items()
                 if current_time - v["ts"] > self.lock_ttl]
        for k in stale:
            del self._locked_riders[k]
            self._pending.pop(k, None)
            self._prev_size.pop(k, None)

    def _lock(self, key, decision: str, conf: float, ts: float) -> None:
        self._locked_riders[key] = {"decision": decision, "conf": conf, "ts": ts}
        self._pending.pop(key, None)
        print(f"Locked {key} -> {decision} ({conf:.0%})")

    @staticmethod
    def _get_rider_key(person_box: tuple,
                       track_id: int = -1,
                       grid: int = 80) -> str | tuple:
        if track_id != -1:
            return f"track_{track_id}"
        cx = (person_box[0] + person_box[2]) // 2
        cy = (person_box[1] + person_box[3]) // 2
        return ((cx // grid) * grid, (cy // grid) * grid)

    @staticmethod
    def _parse_detections(general_detections: list,
                          helmet_detections: list
                          ) -> tuple[list, list, list, list]:
        persons        = []
        motorcycles    = []
        with_helmet    = []
        without_helmet = []

        for det in general_detections:
            if det["conf"] < GENERAL_CONF:
                continue
            if det["class"] == 0:
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

        return persons, motorcycles, with_helmet, without_helmet

    @staticmethod
    def _find_riders(persons: list, motorcycles: list) -> list:
        riders = []
        for person in persons:
            for bike in motorcycles:
                if _is_rider(person["box"], bike):
                    riders.append(person)
                    break
        return riders


# ── Pure functions ────────────────────────────────────────────────────────────

def _box_center(box: tuple) -> tuple:
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def _center_distance(box1: tuple, box2: tuple) -> float:
    c1 = _box_center(box1)
    c2 = _box_center(box2)
    return ((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2) ** 0.5


def _assign_boxes_to_riders(helmet_boxes: list,
                             riders: list) -> dict[int, list]:
    """
    Assign each helmet box exclusively to its nearest rider.

    With 3-5 riders and full-body helmet boxes, a naive approach of
    checking all boxes against each rider causes cross-rider contamination
    — rider A's helmet box overlaps rider B's person_region and gets
    counted for rider B.

    This function builds a 1-to-1 assignment: each helmet box goes to
    exactly one rider (the closest one by center distance). Riders that
    receive no boxes get an empty list.

    Returns: {rider_index: [assigned helmet boxes]}
    """
    assignment: dict[int, list] = {i: [] for i in range(len(riders))}

    for hbox in helmet_boxes:
        if not riders:
            break
        best_idx  = -1
        best_dist = float("inf")
        hcx, hcy  = _box_center(hbox)

        for i, rider in enumerate(riders):
            rx1, ry1, rx2, ry2 = rider["box"]
            # Only consider riders whose person box actually overlaps
            # vertically with the helmet box — rules out riders on
            # completely different rows (e.g. far left vs far right)
            if not (ry1 <= hcy <= ry2 + (ry2-ry1)*0.3):
                continue
            dist = _center_distance(hbox, rider["box"])
            if dist < best_dist:
                best_dist = dist
                best_idx  = i

        if best_idx != -1:
            assignment[best_idx].append(hbox)

    return assignment


def _is_rider(person_box: tuple, bike_box: tuple) -> bool:
    px1, py1, px2, py2 = person_box
    bx1, by1, bx2, by2 = bike_box

    ph       = py2 - py1
    p_cutoff = int(py1 + ph * 0.50)
    p_bottom = (px1, p_cutoff, px2, py2)

    ratio  = compute_overlap_ratio(p_bottom, bike_box)
    simple = boxes_overlap(person_box, bike_box)

    if not (ratio >= 0.20 or simple):
        return False

    person_cx = (px1 + px2) / 2
    bike_cx   = (bx1 + bx2) / 2
    bike_w    = bx2 - bx1
    if abs(person_cx - bike_cx) > bike_w * 0.8:
        return False

    return True


def _get_best_match(head_box:     tuple,
                    helmet_boxes: list,
                    person_box:   tuple,
                    rider_h:      int = 80,
                    is_growing:   bool = False) -> float:
    """
    Score how well any of the assigned helmet boxes matches this rider.
    helmet_boxes here are already pre-assigned to this rider only,
    so no person_region filter needed — we trust the assignment.
    """
    if not helmet_boxes:
        return 0.0

    x1, y1, x2, y2 = head_box
    head_h = y2 - y1
    head_w = x2 - x1

    px1, py1, px2, py2 = person_box
    ph = py2 - py1

    pad = int(head_w * 0.25) if head_h < 30 else int(head_w * 0.08)
    expanded = (
        max(0, x1 - pad),
        max(0, y1 - pad),
        x2 + pad,
        y2 + pad,
    )

    # Upper body weight reduced for close riders
    if rider_h >= CLOSE_RIDER_H:
        ub_iou_w, ub_ov_w = 0.50, 0.45
    else:
        ub_iou_w, ub_ov_w = 0.75, 0.65

    upper_body = (px1, py1, px2, int(py1 + ph * 0.60))

    best = 0.0
    for hbox in helmet_boxes:
        iou         = compute_iou(head_box, hbox)
        overlap     = compute_overlap_ratio(head_box, hbox)
        exp_iou     = compute_iou(expanded, hbox)
        exp_overlap = compute_overlap_ratio(expanded, hbox)
        ub_iou      = compute_iou(upper_body, hbox)
        ub_overlap  = compute_overlap_ratio(upper_body, hbox)

        score = max(iou,             overlap,
                    exp_iou  * 0.8,  exp_overlap  * 0.7,
                    ub_iou   * ub_iou_w, ub_overlap * ub_ov_w)

        if score > best:
            best = score

    return best


def _make_decision(with_score:    float,
                   without_score: float,
                   threshold:     float = HELMET_DOMINANCE_THRESHOLD
                   ) -> tuple[str | None, float]:
    if with_score == 0.0 and without_score == 0.0:
        return None, 0.0

    if with_score >= MIN_SCORE:
        if without_score > with_score * threshold:
            return "violation", without_score
        return "safe", with_score

    if without_score >= MIN_SCORE:
        return "violation", without_score

    return None, 0.0