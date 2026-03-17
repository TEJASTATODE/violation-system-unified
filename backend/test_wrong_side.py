import cv2
import os
from datetime import datetime
from detection.detect import detect_general_objects
from violations.wrong_side_violation import (
    wrong_side_violation,
    reset_wrong_side,
)
from utils.lane_utils import (
    draw_zones,
    draw_vehicle_box,
    draw_status_panel,
    draw_violation_alert,
    get_smooth_center
)

# ── Configuration ─────────────────────────────────────
VIDEO_PATH   = r"C:\Users\TEJAS\OneDrive\Desktop\Miniproject\test13.mp4"
FRAME_SKIP   = 2  # Process detection every N frames
EVIDENCE_DIR = "evidence/wrong_side"
os.makedirs(EVIDENCE_DIR, exist_ok=True)

# ── State ─────────────────────────────────────────────
saved_violations = set()
last_general      = []
last_violations   = []
last_all_vehicles = []

def save_evidence(frame, box, track_id):
    """Saves a cropped image of the violation."""
    if track_id in saved_violations:
        return
    
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, box)
    
    # Add padding for better context in the crop
    pad = 30
    cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
    cx2, cy2 = min(w, x2 + pad), min(h, y2 + pad)
    
    crop = frame[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{EVIDENCE_DIR}/wrongside_id{track_id}_{timestamp}.jpg"
    cv2.imwrite(filename, crop)
    saved_violations.add(track_id)
    print(f"📸 Violation Captured: {filename}")

# ── Processing Loop ───────────────────────────────────
cap = cv2.VideoCapture(VIDEO_PATH)
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    h, w = frame.shape[:2]

    # 1. Update Road Geometry (Always update for smooth UI)
    center_x = get_smooth_center(frame)

    # 2. Heavy Inference (Skipped frames)
    if frame_count % FRAME_SKIP == 0:
        # Detect objects
        last_general = detect_general_objects(frame)

        # Process violations with our improved logic
        last_violations, last_all_vehicles = wrong_side_violation(
            last_general,
            frame_width = w,
            center_x    = center_x
        )

        # Handle evidence saving
        for v in last_violations:
            save_evidence(frame, v["box"], v["track_id"])

    # 3. Visualization (Every frame)
    # Draw Lane Zones
    draw_zones(frame, center_x)

    # Draw Boxes and Labels
    # Note: 'approaching' is now calculated inside wrong_side_violation
    for v in last_all_vehicles:
        draw_vehicle_box(
            frame,
            box         = v["box"],
            track_id    = v["track_id"],
            zone        = v["zone"],
            approaching = v.get("dy", 0) > 1.0, # Moving toward camera
            violation   = v["violation"]
        )

    # UI Overlay
    num_approaching = sum(1 for v in last_all_vehicles if v.get("dy", 0) > 1.0)
    
    draw_status_panel(
        frame,
        total       = len(last_all_vehicles),
        approaching = num_approaching,
        violations  = len(last_violations)
    )

    if len(last_violations) > 0:
        draw_violation_alert(frame, len(last_violations))

    # 4. Display & Controls
    display = cv2.resize(frame, (1080, 600))
    cv2.imshow("Smart Traffic Monitor - Wrong Side Detection", display)

    key = cv2.waitKey(1) & 0xFF
    if key == 27: # ESC
        break
    elif key == 32: # SPACE (Pause)
        cv2.waitKey(0)
    elif key == ord('r'): # Reset
        reset_wrong_side()
        saved_violations.clear()
        print("🔄 System Reset!")

cap.release()
cv2.destroyAllWindows()