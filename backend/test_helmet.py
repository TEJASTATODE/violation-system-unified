import cv2
import os
import time
from datetime import datetime
from detection.detect import detect_general_objects, detect_helmet_objects
from violations.helmet_violation import helmet_violation, reset_locked_riders
from utils.draw import draw_box, draw_status_panel, draw_violation_alert

video_path = r"C:\Users\TEJAS\OneDrive\Desktop\Miniproject\test6.mp4"

os.makedirs("evidence/violations", exist_ok=True)

cap             = cv2.VideoCapture(video_path)
FRAME_SKIP      = 3
frame_count     = 0
last_general    = []
last_helmet     = []
last_violations = []
last_safe       = []
saved_riders    = set()

# ── FPS tracking ─────────────────────────────────────
fps          = 0
fps_counter  = 0
fps_start    = time.time()


def rider_key(box, track_id=-1, grid=100):
    if track_id != -1:
        return f"track_{track_id}"
    cx = (box[0]+box[2])//2
    cy = (box[1]+box[3])//2
    return ((cx//grid)*grid, (cy//grid)*grid)


def save_crop(frame, rider_box, bike_box=None):
    h, w = frame.shape[:2]
    if bike_box:
        cx1 = min(rider_box[0], bike_box[0])
        cy1 = min(rider_box[1], bike_box[1])
        cx2 = max(rider_box[2], bike_box[2])
        cy2 = max(rider_box[3], bike_box[3])
    else:
        cx1, cy1, cx2, cy2 = rider_box
    pad = 15
    cx1 = max(0, cx1-pad)
    cy1 = max(0, cy1-pad)
    cx2 = min(w,  cx2+pad)
    cy2 = min(h,  cy2+pad)
    crop = frame[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = f"evidence/violations/violation_{ts}.jpg"
    cv2.imwrite(path, crop)
    print(f"🚨 Saved: {path}")


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # ── Calculate FPS ─────────────────────────────────
    fps_counter += 1
    elapsed = time.time() - fps_start
    if elapsed >= 1.0:
        fps       = fps_counter / elapsed
        fps_counter = 0
        fps_start   = time.time()

    if frame_count % FRAME_SKIP == 0:
        last_general    = detect_general_objects(frame)
        last_helmet     = detect_helmet_objects(frame)
        last_violations, last_safe = helmet_violation(
            last_general, last_helmet
        )

        for v in last_violations:
            key = rider_key(
                v["rider_box"],
                track_id=v.get("track_id", -1)
            )
            if key not in saved_riders:
                bike_box  = None
                best_dist = float("inf")
                rx = (v["rider_box"][0]+v["rider_box"][2])//2
                ry = (v["rider_box"][1]+v["rider_box"][3])//2
                for det in last_general:
                    if det["class"] == 3:
                        bx = (det["box"][0]+det["box"][2])//2
                        by = (det["box"][1]+det["box"][3])//2
                        dist = abs(rx-bx)+abs(ry-by)
                        if dist < best_dist:
                            best_dist = dist
                            bike_box  = det["box"]
                save_crop(frame, v["rider_box"], bike_box)
                saved_riders.add(key)

    # RED = violation
    for v in last_violations:
        draw_box(frame, v["rider_box"],
                 color=(0,0,255),
                 label="NO HELMET",
                 thickness=2)

    # YELLOW = safe
    for s in last_safe:
        draw_box(frame, s["rider_box"],
                 color=(0,255,255),
                 label="HELMET OK",
                 thickness=2)

    # Status panel
    draw_status_panel(
        frame,
        persons    = sum(1 for d in last_general if d["class"]==0),
        bikes      = sum(1 for d in last_general if d["class"]==3),
        helmets    = len(last_helmet),
        violations = len(last_violations),
        safe       = len(last_safe)
    )

    draw_violation_alert(frame, len(last_violations))

    # ── Show FPS on frame ─────────────────────────────
    cv2.putText(frame, f"FPS: {fps:.1f}",
                (frame.shape[1]-120, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0,255,0), 2)

    cv2.imshow("Helmet Detection", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == 32:
        cv2.waitKey(0)
    elif key == ord('r'):
        reset_locked_riders()
        saved_riders.clear()
        print("🔄 Reset!")

cap.release()
cv2.destroyAllWindows()
