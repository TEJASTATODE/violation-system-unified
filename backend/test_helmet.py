import cv2
import time
from detection.detect import detect_general_objects, detect_helmet_objects
from violations.helmet_violation import HelmetViolationDetector
from utils.evidence import EvidenceRecorder, ensure_evidence_dirs
from utils.draw import blend_rect

# ── Config ────────────────────────────────────────────────────────────────────
VIDEO_PATH   = r"C:\Users\TEJAS\OneDrive\Desktop\Miniproject\test_videos\test20.mp4"
FRAME_SKIP   = 3

DISPLAY_HOLD = 8

# ── Evidence ──────────────────────────────────────────────────────────────────
# Uses the same shared EvidenceRecorder as run_pipeline.py (utils/evidence.py)
# instead of ad-hoc save logic, so isolated-detector runs land in the same
# backend/evidence/helmet/ folder as the unified pipeline — one place, not two.
ensure_evidence_dirs()
recorder = EvidenceRecorder("helmet")

# ── Open video ────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# ── Detector ──────────────────────────────────────────────────────────────────
detector = HelmetViolationDetector(frame_w=frame_w, frame_h=frame_h)

# ── State ─────────────────────────────────────────────────────────────────────
frame_count     = 0
last_general    = []
last_helmet     = []
last_violations = []
last_safe       = []

fps         = 0.0
fps_counter = 0
fps_start   = time.time()


display_boxes: list[dict] = []

VIOLATION_COLOR = (0,   0,   255)
SAFE_COLOR      = (0,   255, 255)

def nearest_bike(rider_box: tuple, general_dets: list) -> tuple | None:
    rx = (rider_box[0] + rider_box[2]) // 2
    ry = (rider_box[1] + rider_box[3]) // 2
    best_box, best_dist = None, float("inf")
    for det in general_dets:
        if det["class"] == 3:
            bx = (det["box"][0] + det["box"][2]) // 2
            by = (det["box"][1] + det["box"][3]) // 2
            d  = abs(rx - bx) + abs(ry - by)
            if d < best_dist:
                best_dist, best_box = d, det["box"]
    return best_box


def update_display_boxes(violations: list, safe_riders: list) -> None:
    """
    Merge fresh detections into the display buffer.
    - New detections reset the TTL to DISPLAY_HOLD.
    - Existing entries count down each call.
    - Violation always overwrites safe for same rider area.
    """
    # Tick down all existing entries
    for db in display_boxes:
        db["ttl"] -= 1

    # Remove expired entries
    display_boxes[:] = [db for db in display_boxes if db["ttl"] > 0]

    def _box_center(b):
        return ((b[0]+b[2])//2, (b[1]+b[3])//2)

    def _close(b1, b2, thresh=60):
        c1, c2 = _box_center(b1), _box_center(b2)
        return abs(c1[0]-c2[0]) < thresh and abs(c1[1]-c2[1]) < thresh

    # Upsert fresh detections into the buffer
    for v in violations:
        matched = False
        for db in display_boxes:
            if _close(db["rider_box"], v["rider_box"]):
                # Always upgrade to violation if same rider
                db.update({"rider_box": v["rider_box"],
                            "head_box":  v["head_box"],
                            "conf":      v["conf"],
                            "label":     "NO HELMET",
                            "color":     VIOLATION_COLOR,
                            "ttl":       DISPLAY_HOLD})
                matched = True
                break
        if not matched:
            display_boxes.append({"rider_box": v["rider_box"],
                                   "head_box":  v["head_box"],
                                   "conf":      v["conf"],
                                   "label":     "NO HELMET",
                                   "color":     VIOLATION_COLOR,
                                   "ttl":       DISPLAY_HOLD})

    for s in safe_riders:
        matched = False
        for db in display_boxes:
            if _close(db["rider_box"], s["rider_box"]):
                # Only update if currently safe — don't downgrade a violation
                if db["label"] != "NO HELMET":
                    db.update({"rider_box": s["rider_box"],
                                "head_box":  s["head_box"],
                                "conf":      s["conf"],
                                "ttl":       DISPLAY_HOLD})
                matched = True
                break
        if not matched:
            display_boxes.append({"rider_box": s["rider_box"],
                                   "head_box":  s["head_box"],
                                   "conf":      s["conf"],
                                   "label":     "HELMET OK",
                                   "color":     SAFE_COLOR,
                                   "ttl":       DISPLAY_HOLD})


OUTLINE_THICKNESS = 2   # cv2.rectangle paints ~1px beyond given coords per side at this thickness (verified empirically)


def draw_boxes(frame) -> None:
    h, w = frame.shape[:2]
    pad = OUTLINE_THICKNESS
    for db in display_boxes:
        x1, y1, x2, y2 = [int(c) for c in db["rider_box"]]
        color = db["color"]
        # Fade box slightly as TTL drops — full opacity at DISPLAY_HOLD,
        # 50% at 1
        alpha = max(0.5, db["ttl"] / DISPLAY_HOLD)

        # Blend only this box's own region instead of a full-frame copy per
        # box per frame — same translucent-outline look, far less work.
        # Pad the crop by the outline thickness — an unpadded crop clips
        # part of the outline's right/bottom edge.
        rx1, ry1 = max(0, x1 - pad), max(0, y1 - pad)
        rx2, ry2 = min(w - 1, x2 + pad), min(h - 1, y2 + pad)
        if rx2 >= rx1 and ry2 >= ry1:
            roi     = frame[ry1:ry2 + 1, rx1:rx2 + 1]
            overlay = roi.copy()
            cv2.rectangle(overlay, (x1 - rx1, y1 - ry1), (x2 - rx1, y2 - ry1), color, OUTLINE_THICKNESS)
            cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, roi)

        cv2.putText(frame, db["label"], (x1, y1-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)


def draw_hud(frame, fps: float, n_viol: int, n_safe: int) -> None:
    lines = [f"FPS: {fps:.1f}",
             f"Violations: {n_viol}",
             f"Safe: {n_safe}"]
    blend_rect(frame, 6, 6, 200, 72, (18, 18, 18), 0.55)
    for i, txt in enumerate(lines):
        cv2.putText(frame, txt, (14, 24+i*18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                    (190, 220, 190), 1, cv2.LINE_AA)


# ── Main loop ─────────────────────────────────────────────────────────────────
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count  += 1
        current_time  = time.time()

        # FPS counter
        fps_counter += 1
        elapsed = current_time - fps_start
        if elapsed >= 1.0:
            fps         = fps_counter / elapsed
            fps_counter = 0
            fps_start   = current_time

        recorder.expire(frame_count)

        # Run detection every FRAME_SKIP frames
        if frame_count % FRAME_SKIP == 0:
            last_general    = detect_general_objects(frame)
            last_helmet     = detect_helmet_objects(frame)
            last_violations, last_safe = detector.process_frame(
                last_general, last_helmet, current_time=current_time
            )

            # Save evidence for new violations
            for v in last_violations:
                bike_box = nearest_bike(v["rider_box"], last_general)
                recorder.maybe_save(frame, v["rider_box"], frame_count,
                                     v.get("track_id", -1), extra_box=bike_box)

        # Update persistent display buffer every frame
        update_display_boxes(last_violations, last_safe)

        # Draw persistent boxes and HUD
        draw_boxes(frame)
        draw_hud(frame, fps,
                 sum(1 for db in display_boxes if db["label"] == "NO HELMET"),
                 sum(1 for db in display_boxes if db["label"] == "HELMET OK"))

        cv2.imshow("Helmet Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:                   # ESC — quit
            break
        elif key == 32:                 # SPACE — pause
            cv2.waitKey(0)
        elif key == ord('r'):           # R — reset
            detector.reset()
            recorder.reset()
            display_boxes.clear()
            print("Reset!")

finally:
    cap.release()
    cv2.destroyAllWindows()