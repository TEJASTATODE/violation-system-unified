import cv2
import os
import time
from datetime import datetime
from detection.detect import detect_general_objects, detect_helmet_objects
from violations.helmet_violation import HelmetViolationDetector

# ── Config ────────────────────────────────────────────────────────────────────
VIDEO_PATH   = r"C:\Users\TEJAS\OneDrive\Desktop\Miniproject\test20.mp4"
EVIDENCE_DIR = "evidence/violations"
FRAME_SKIP   = 3


DISPLAY_HOLD        = 8
SAVED_EXPIRY_FRAMES = 150

os.makedirs(EVIDENCE_DIR, exist_ok=True)

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
saved_riders: dict[str, int] = {}

fps         = 0.0
fps_counter = 0
fps_start   = time.time()


display_boxes: list[dict] = []

VIOLATION_COLOR = (0,   0,   255)
SAFE_COLOR      = (0,   255, 255)

def rider_key(box: tuple, track_id: int = -1, grid: int = 100) -> str:
    if track_id != -1:
        return f"track_{track_id}"
    cx = (box[0] + box[2]) // 2
    cy = (box[1] + box[3]) // 2
    return f"grid_{(cx // grid) * grid}_{(cy // grid) * grid}"

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


def save_crop(frame, rider_box: tuple, bike_box: tuple | None) -> None:
    h, w = frame.shape[:2]
    if bike_box:
        cx1 = min(rider_box[0], bike_box[0])
        cy1 = min(rider_box[1], bike_box[1])
        cx2 = max(rider_box[2], bike_box[2])
        cy2 = max(rider_box[3], bike_box[3])
    else:
        cx1, cy1, cx2, cy2 = rider_box
    pad = 15
    cx1 = max(0, cx1-pad);  cy1 = max(0, cy1-pad)
    cx2 = min(w, cx2+pad);  cy2 = min(h, cy2+pad)
    crop = frame[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = os.path.join(EVIDENCE_DIR, f"violation_{ts}.jpg")
    cv2.imwrite(path, crop)
    print(f"Saved: {path}")


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


def draw_boxes(frame) -> None:
    for db in display_boxes:
        x1, y1, x2, y2 = [int(c) for c in db["rider_box"]]
        color = db["color"]
        # Fade box slightly as TTL drops — full opacity at DISPLAY_HOLD,
        # 50% at 1
        alpha = max(0.5, db["ttl"] / DISPLAY_HOLD)
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)
        cv2.putText(frame, db["label"], (x1, y1-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)


def draw_hud(frame, fps: float, n_viol: int, n_safe: int) -> None:
    lines = [f"FPS: {fps:.1f}",
             f"Violations: {n_viol}",
             f"Safe: {n_safe}"]
    overlay = frame.copy()
    cv2.rectangle(overlay, (6, 6), (200, 72), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
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

        # Expire saved rider keys
        for k in list(saved_riders.keys()):
            if frame_count - saved_riders[k] > SAVED_EXPIRY_FRAMES:
                del saved_riders[k]

        # Run detection every FRAME_SKIP frames
        if frame_count % FRAME_SKIP == 0:
            last_general    = detect_general_objects(frame)
            last_helmet     = detect_helmet_objects(frame)
            last_violations, last_safe = detector.process_frame(
                last_general, last_helmet, current_time=current_time
            )

            # Save evidence for new violations
            for v in last_violations:
                key = rider_key(v["rider_box"], track_id=v.get("track_id", -1))
                if key not in saved_riders:
                    bike_box = nearest_bike(v["rider_box"], last_general)
                    save_crop(frame, v["rider_box"], bike_box)
                    saved_riders[key] = frame_count

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
            saved_riders.clear()
            display_boxes.clear()
            print("Reset!")

finally:
    cap.release()
    cv2.destroyAllWindows()