import cv2
import time

from detection.detect import (
    detect_general_objects,
    detect_helmet_objects,
    detect_triple_riding_objects,
    detect_signal_objects,
    detect_smoke_objects,
)
from violations.wrong_side_violation import process_frame as wrong_side_process, reset_wrong_side_state
from violations.helmet_violation import HelmetViolationDetector
from violations.triple_riding_violation import TripleRidingViolationDetector
from violations.signal_jump_violation import SignalJumpingDetector
from violations.smoke_emission_violation import SmokeEmissionDetector
from utils.lane_utils import draw_roi
from utils.draw import draw_box, blend_rect
from utils.evidence import EvidenceRecorder, ensure_evidence_dirs

# ── Config ────────────────────────────────────────────────────────────────
VIDEO_PATH = r"C:\Users\TEJAS\OneDrive\Desktop\Miniproject\test_videos\test20.mp4"
FRAME_SKIP = 3          # applies to helmet + triple-riding + smoke-emission only (see plan notes)

DISPLAY_HOLD = 8        # frames a helmet box stays visible between detection calls

TRIPLE_COLOR = (255, 0, 255)   # magenta
SIGNAL_COLOR = (0, 0, 255)     # red — signal-jump violation box
SMOKE_COLOR  = (0, 140, 255)   # orange — confirmed smoke-emission violation box

# ── Setup ─────────────────────────────────────────────────────────────────
ensure_evidence_dirs()
recorders = {
    "wrong_side":     EvidenceRecorder("wrong_side"),
    "helmet":         EvidenceRecorder("helmet"),
    "triple_riding":  EvidenceRecorder("triple_riding"),
    "signal_jump":    EvidenceRecorder("signal_jump"),
    "smoke_emission": EvidenceRecorder("smoke_emission"),
}

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

helmet_detector = HelmetViolationDetector(frame_w=frame_w, frame_h=frame_h)
triple_detector = TripleRidingViolationDetector()
signal_detector = SignalJumpingDetector()
smoke_detector  = SmokeEmissionDetector()

prev_frame  = None
frame_count = 0

last_general     = []
last_helmet_viol = []
last_helmet_safe = []
last_triple_viol = []
last_signal_viol = []
signal_color     = "unknown"
signal_stop_y    = 0
signal_buffer_y  = 0
last_smoke_viol  = []

display_boxes: list[dict] = []   # helmet's TTL-fade display buffer

fps         = 0.0
fps_counter = 0
fps_start   = time.time()

VIOLATION_COLOR = (0,   0, 255)
SAFE_COLOR      = (0, 255, 255)

# ── Fault isolation ──────────────────────────────────────────────────────
# One detector's edge-case failure (bad frame, model error, ...) must not
# take down the other four in a system meant to run unattended in a car.
ERROR_LOG_COOLDOWN = 5.0   # seconds between repeated error logs per detector
_last_error_log: dict = {}


def safe_call(name: str, fn, fallback):
    """Runs fn(); on exception, logs (rate-limited) and returns fallback
    (typically the previous frame's output) so the pipeline keeps going."""
    try:
        return fn()
    except Exception as e:
        now = time.time()
        if now - _last_error_log.get(name, 0) > ERROR_LOG_COOLDOWN:
            print(f"[ERROR] {name} detector failed this frame: {e!r} (using last-known output)")
            _last_error_log[name] = now
        return fallback


def nearest_bike(rider_box: tuple, general_dets: list):
    rx = (rider_box[0] + rider_box[2]) // 2
    ry = (rider_box[1] + rider_box[3]) // 2
    best_box, best_dist = None, float("inf")
    for det in general_dets:
        if det["class"] == 3:   # COCO motorcycle
            bx = (det["box"][0] + det["box"][2]) // 2
            by = (det["box"][1] + det["box"][3]) // 2
            d  = abs(rx - bx) + abs(ry - by)
            if d < best_dist:
                best_dist, best_box = d, det["box"]
    return best_box


def update_display_boxes(violations: list, safe_riders: list) -> None:
    for db in display_boxes:
        db["ttl"] -= 1
    display_boxes[:] = [db for db in display_boxes if db["ttl"] > 0]

    def _center(b):
        return ((b[0] + b[2]) // 2, (b[1] + b[3]) // 2)

    def _close(b1, b2, thresh=60):
        c1, c2 = _center(b1), _center(b2)
        return abs(c1[0] - c2[0]) < thresh and abs(c1[1] - c2[1]) < thresh

    for v in violations:
        matched = False
        for db in display_boxes:
            if _close(db["box"], v["rider_box"]):
                db.update({"box": v["rider_box"], "label": "NO HELMET",
                           "color": VIOLATION_COLOR, "ttl": DISPLAY_HOLD})
                matched = True
                break
        if not matched:
            display_boxes.append({"box": v["rider_box"], "label": "NO HELMET",
                                   "color": VIOLATION_COLOR, "ttl": DISPLAY_HOLD})

    for s in safe_riders:
        matched = False
        for db in display_boxes:
            if _close(db["box"], s["rider_box"]):
                if db["label"] != "NO HELMET":
                    db.update({"box": s["rider_box"], "ttl": DISPLAY_HOLD})
                matched = True
                break
        if not matched:
            display_boxes.append({"box": s["rider_box"], "label": "HELMET OK",
                                   "color": SAFE_COLOR, "ttl": DISPLAY_HOLD})


def draw_display_boxes(frame) -> None:
    for db in display_boxes:
        draw_box(frame, [int(c) for c in db["box"]], color=db["color"], label=db["label"])


def draw_combined_hud(frame, fps: float, counts: dict) -> None:
    lines = [
        f"FPS: {fps:.1f}",
        f"Wrong-side: {counts['wrong_side']}",
        f"No helmet: {counts['helmet']}",
        f"Triple riding: {counts['triple_riding']}",
        f"Signal jump: {counts['signal_jump']}",
        f"Smoke emission: {counts['smoke_emission']}",
    ]
    blend_rect(frame, frame.shape[1] - 210, 6, frame.shape[1] - 6, 6 + 20 * len(lines) + 10,
               (18, 18, 18), 0.55)
    for i, txt in enumerate(lines):
        cv2.putText(frame, txt, (frame.shape[1] - 202, 24 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (190, 220, 190), 1, cv2.LINE_AA)


# ── Main loop ────────────────────────────────────────────────────────────
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        now = time.time()

        fps_counter += 1
        elapsed = now - fps_start
        if elapsed >= 1.0:
            fps = fps_counter / elapsed
            fps_counter = 0
            fps_start = now

        if prev_frame is None:
            prev_frame = frame.copy()
            continue

        raw_frame = frame.copy()   # clean pixels for evidence/optical-flow continuity
        for rec in recorders.values():
            rec.expire(frame_count)

        # ── every frame: general detection + wrong-side + signal-jump (need continuous optical flow) ──
        last_general = safe_call("general_detect", lambda: detect_general_objects(frame), [])

        canvas, wrong_side_viol = safe_call(
            "wrong_side",
            lambda: wrong_side_process(frame, prev_frame, general_dets=last_general),
            (frame.copy(), []),
        )
        for v in wrong_side_viol:
            recorders["wrong_side"].maybe_save(raw_frame, v["box"], frame_count, v["track_id"])

        signal_dets = safe_call("signal_detect", lambda: detect_signal_objects(frame), [])
        last_signal_viol, signal_color, signal_stop_y, signal_buffer_y, _signal_all = safe_call(
            "signal_jump",
            lambda: signal_detector.process_frame(frame, signal_dets, current_time=now),
            (last_signal_viol, signal_color, signal_stop_y, signal_buffer_y, {}),
        )
        for v in last_signal_viol:
            recorders["signal_jump"].maybe_save(raw_frame, v["bbox"], frame_count, v["tid"])

        # ── every FRAME_SKIP frames: helmet + triple-riding + smoke-emission (tolerant to gaps) ──
        if frame_count % FRAME_SKIP == 0:
            helmet_dets = safe_call("helmet_detect", lambda: detect_helmet_objects(frame), [])
            last_helmet_viol, last_helmet_safe = safe_call(
                "helmet",
                lambda: helmet_detector.process_frame(last_general, helmet_dets, current_time=now),
                (last_helmet_viol, last_helmet_safe),
            )
            for v in last_helmet_viol:
                bike = nearest_bike(v["rider_box"], last_general)
                recorders["helmet"].maybe_save(raw_frame, v["rider_box"], frame_count,
                                                v.get("track_id", -1), extra_box=bike)

            triple_dets = safe_call("triple_detect", lambda: detect_triple_riding_objects(frame), [])
            last_triple_viol = safe_call(
                "triple_riding",
                lambda: triple_detector.process_frame(triple_dets, current_time=now),
                last_triple_viol,
            )
            for v in last_triple_viol:
                recorders["triple_riding"].maybe_save(raw_frame, v["box"], frame_count, v["track_id"])

            smoke_dets = safe_call("smoke_detect", lambda: detect_smoke_objects(frame), [])
            last_smoke_viol = safe_call(
                "smoke_emission",
                lambda: smoke_detector.process_frame(smoke_dets, current_time=now),
                last_smoke_viol,
            )
            for v in last_smoke_viol:
                recorders["smoke_emission"].maybe_save(raw_frame, v["box"], frame_count, v.get("track_id", -1))

        update_display_boxes(last_helmet_viol, last_helmet_safe)

        # ── draw, in order (wrong_side's canvas is the base) ──
        draw_display_boxes(canvas)
        for v in last_triple_viol:
            draw_box(canvas, v["box"], color=TRIPLE_COLOR, label="TRIPLE RIDING")

        # signal-jump: stop/buffer lines + light-colour indicator + violation boxes
        cv2.line(canvas, (0, signal_stop_y),   (frame_w, signal_stop_y),   (0, 255, 255), 2)
        cv2.line(canvas, (0, signal_buffer_y), (frame_w, signal_buffer_y), (0, 165, 255), 1)
        sig_bgr = (0, 0, 220)   if signal_color == "red"   else \
                  (0, 200, 0)   if signal_color == "green" else \
                  (180, 180, 0)
        cv2.rectangle(canvas, (10, 10), (220, 46), (30, 30, 30), -1)
        cv2.putText(canvas, f"LIGHT: {signal_color.upper()}",
                    (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, sig_bgr, 2)
        for v in last_signal_viol:
            draw_box(canvas, v["bbox"], color=SIGNAL_COLOR, label=f"SIGNAL JUMP ID:{v['tid']}")

        for v in last_smoke_viol:
            draw_box(canvas, v["box"], color=SMOKE_COLOR, label="SMOKE EMISSION")

        canvas = draw_roi(canvas)
        draw_combined_hud(canvas, fps, {
            "wrong_side":     len(wrong_side_viol),
            "helmet":         sum(1 for db in display_boxes if db["label"] == "NO HELMET"),
            "triple_riding":  len(last_triple_viol),
            "signal_jump":    len(last_signal_viol),
            "smoke_emission": len(last_smoke_viol),
        })

        cv2.imshow("Unified Dashcam Pipeline", canvas)
        prev_frame = raw_frame

        key = cv2.waitKey(1) & 0xFF
        if key == 27:                    # ESC — quit
            break
        elif key == 32:                  # SPACE — pause
            cv2.waitKey(0)
        elif key == ord('r'):            # R — reset all detectors
            reset_wrong_side_state()
            helmet_detector.reset()
            triple_detector.reset()
            signal_detector.reset()
            smoke_detector.reset()
            for rec in recorders.values():
                rec.reset()
            display_boxes.clear()
            print("Reset!")

finally:
    cap.release()
    cv2.destroyAllWindows()
