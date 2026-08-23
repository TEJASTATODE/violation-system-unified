import cv2
import numpy as np
from time import time


class SignalJumpingDetector:
    """
    Detects signal-jump violations from a front-mounted dashcam.

    Violation logic — ALL three gates must pass in sequence:
        Gate 1  →  Signal is RED  (smoothed over SIG_CONFIRM frames)
        Gate 2  →  Vehicle was CONFIRMED STOPPED (still for MIN_STILL_FRAMES)
        Gate 3  →  Vehicle then MOVED (centroid shifted ≥ JUMP_PX from locked position)

    A standing car satisfies Gate 1 + Gate 2 but never Gate 3 → zero false positives.

    Ported from Singnal-Jump-Detector/detectors/signal_detector.py. The YOLO call that
    used to happen inside process_frame() now happens in detection/yolo_model.py +
    detection/detect.py (detect_signal_objects), matching how helmet/triple-riding
    detectors receive pre-computed detections. All gate thresholds and the optical-flow
    stillness check are unchanged.

    Caveat: thresholds below (LK_STILL_THRESH, JUMP_PX) were tuned against frames
    resized to 960x540 by the original standalone script. run_pipeline.py runs at
    native video resolution, so behaviour may shift slightly at other resolutions.
    Not retuned here.

    Known limitation (not fixable by threshold tuning): Gate 1 needs SIG_CONFIRM
    consecutive red-light detections before it ever declares "red". Measured on
    this repo's test*.mp4 clips, the model (detect_signal_objects) sees ~800
    vehicle boxes per ~150 frames but only 0-3 red/green light boxes total —
    none of the test footage is actually filmed at a visible traffic light, so
    Gate 1 is structurally unreachable on it. imgsz was raised to 960 and the
    light classes get a lower confidence bar (detection/detect.py) to catch
    what's genuinely there, but no amount of tuning substitutes for footage
    that shows a red light. Real validation needs an intersection clip.
    """

    # ── tunables ──────────────────────────────────────────────────────────
    STOP_LINE_RATIO   = 0.65   # stop line at 65% of frame height
    LINE_BUFFER       = 40     # px below stop line = bottom of zone

    LK_STILL_THRESH   = 1.8    # LK magnitude below this → vehicle is still
                               # raise to 2.5–3.0 if your camera vibrates a lot
    MIN_STILL_FRAMES  = 20     # consecutive still frames to confirm stopped (~0.7 s @ 30fps)
    JUMP_PX           = 40     # centroid must shift this many px to be a "jump"

    SIG_CONFIRM       = 8      # consecutive red detections  → declare RED
    SIG_LOSE          = 10     # consecutive non-red frames  → clear RED

    VIOLATION_COOLDOWN = 60    # frames to ignore a vehicle after capturing it

    VEHICLE_CLS = 0            # class id for vehicle in signal.pt
    RED_CLS     = 1            # class id for red  light
    GREEN_CLS   = 2            # class id for green light

    TRACK_STALE_SEC = 10.0     # forget a vehicle's state if unseen this long

    # ─────────────────────────────────────────────────────────────────────

    def __init__(self):
        # optical flow
        self.prev_gray = None

        # per-vehicle state  {tid: value}
        self._still_count       = {}
        self._confirmed_stopped = {}
        self._stopped_centroid  = {}   # (cx, cy) locked when car confirmed stopped
        self._violation_logged  = {}
        self._cooldown          = {}
        self._prev_centroid     = {}   # for debugging only
        self._last_seen         = {}   # tid -> ts, for reaping vehicles that left frame

        # signal smoother
        self._red_streak   = 0
        self._green_streak = 0
        self._signal_state = "unknown"

    # ──────────────────────────────────────────────────────────────────────
    #  INTERNAL HELPERS
    # ──────────────────────────────────────────────────────────────────────

    def _smooth_signal(self, raw_red: bool, raw_green: bool) -> str:
        """
        Returns the smoothed signal colour.
        Needs SIG_CONFIRM consecutive red   detections to declare 'red'.
        Needs SIG_LOSE   consecutive !red   frames      to clear  'red'.
        Holds state when nothing is detected (handles brief occlusions).
        """
        if raw_red:
            self._red_streak   += 1
            self._green_streak  = 0
        elif raw_green:
            self._green_streak += 1
            self._red_streak    = 0
        # no detection → hold both streaks (intentional)

        if self._red_streak   >= self.SIG_CONFIRM:
            self._signal_state = "red"
        if self._green_streak >= self.SIG_LOSE:
            self._signal_state = "green"

        return self._signal_state

    def _init_vehicle(self, tid):
        """Create default state for a newly seen vehicle ID."""
        self._still_count[tid]       = 0
        self._confirmed_stopped[tid] = False
        self._stopped_centroid[tid]  = None
        self._violation_logged[tid]  = False
        self._cooldown[tid]          = 0
        self._prev_centroid[tid]     = None

    def _reset_stop_state(self, tid):
        """
        Called when light turns green or vehicle leaves the zone.
        Clears the stop cycle so the vehicle can be re-evaluated next red phase.
        Does NOT reset violation_logged mid-cooldown.
        """
        self._still_count[tid]       = 0
        self._confirmed_stopped[tid] = False
        self._stopped_centroid[tid]  = None
        if self._cooldown[tid] == 0:          # only reset flag when cooldown done
            self._violation_logged[tid] = False

    def _compute_lk_magnitude(self, prev_gray, curr_gray, bbox) -> float:
        """
        Lucas-Kanade optical flow over the vehicle ROI.
        Returns mean magnitude of all tracked point displacements.
        Uses full 2-D magnitude (sqrt(dx²+dy²)), not just dy.
        """
        x1, y1, x2, y2 = map(int, bbox)
        roi_prev = prev_gray[y1:y2, x1:x2]
        roi_curr = curr_gray[y1:y2, x1:x2]

        if roi_prev.size == 0 or roi_curr.size == 0:
            return 0.0

        p0 = cv2.goodFeaturesToTrack(roi_prev, maxCorners=15,
                                     qualityLevel=0.3, minDistance=5)
        if p0 is None:
            return 0.0

        p1, status, _ = cv2.calcOpticalFlowPyrLK(roi_prev, roi_curr, p0, None)
        if p1 is None:
            return 0.0

        good_new = p1[status.ravel() == 1].reshape(-1, 2)   # (N,1,2) → (N,2)
        good_old = p0[status.ravel() == 1].reshape(-1, 2)
        if len(good_new) == 0:
            return 0.0

        delta = good_new - good_old                          # now safely (N,2)
        magnitudes = np.sqrt(delta[:, 0] ** 2 + delta[:, 1] ** 2)
        return float(np.mean(magnitudes))

    # ──────────────────────────────────────────────────────────────────────
    #  MAIN PUBLIC METHOD
    # ──────────────────────────────────────────────────────────────────────

    def process_frame(self, frame, signal_dets, current_time=None):
        """
        Process one frame.

        Parameters
        ----------
        frame       : current BGR frame (needed for optical flow)
        signal_dets : list of dicts from detect_signal_objects(frame)
                      {"box": (x1,y1,x2,y2), "track_id": int, "conf": float, "class": int}

        Returns
        -------
        violations   : list of dicts  {"bbox": [x1,y1,x2,y2], "tid": int}
        light_color  : str  "red" | "green" | "unknown"
        stop_line_y  : int  pixel Y of stop line
        buffer_line_y: int  pixel Y of buffer line
        detections   : dict {"vehicles": [...], "red_lights": [...], "green_lights": [...]}
        """
        if current_time is None:
            current_time = time()

        self._expire_stale(current_time)

        h, w = frame.shape[:2]

        # ── 1. Split pre-computed detections by class ──────────────────
        vehicles     = []
        red_lights   = []
        green_lights = []

        for det in signal_dets:
            cls  = det["class"]
            bbox = list(det["box"])
            obj  = {"bbox": bbox, "cls": cls, "track_id": det.get("track_id", -1)}
            if   cls == self.VEHICLE_CLS: vehicles.append(obj)
            elif cls == self.RED_CLS:     red_lights.append(obj)
            elif cls == self.GREEN_CLS:   green_lights.append(obj)

        # ── 2. Smooth signal colour ────────────────────────────────────
        light_color = self._smooth_signal(
            raw_red   = len(red_lights)   > 0,
            raw_green = len(green_lights) > 0,
        )

        # ── 3. Zone geometry ───────────────────────────────────────────
        stop_line_y   = int(h * self.STOP_LINE_RATIO)
        buffer_line_y = stop_line_y + self.LINE_BUFFER

        # ── 4. Optical flow frame ──────────────────────────────────────
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ── 5. Per-vehicle state machine ───────────────────────────────
        violations = []

        for v in vehicles:
            tid = v["track_id"]
            if tid == -1:
                continue   # signal_model.track() failed to assign an id this frame

            bbox = v["bbox"]
            x1, y1, x2, y2 = bbox

            self._last_seen[tid] = current_time

            # initialise state for new IDs
            if tid not in self._still_count:
                self._init_vehicle(tid)

            # centroid (stable; avoids YOLO bbox jitter of ±4–6 px on y2)
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # ── cooldown tick ──────────────────────────────────────────
            if self._cooldown[tid] > 0:
                self._cooldown[tid] -= 1
                self._prev_centroid[tid] = (cx, cy)
                continue

            # ── LK magnitude ──────────────────────────────────────────
            lk_mag = 0.0
            if self.prev_gray is not None:
                lk_mag = self._compute_lk_magnitude(self.prev_gray, curr_gray, bbox)

            # ── zone check ────────────────────────────────────────────
            # centroid must be inside the stop zone (between stop line and buffer+margin)
            in_stop_zone = (stop_line_y <= cy <= buffer_line_y + 60)

            # ── gate 0: must be red AND in zone ───────────────────────
            if light_color != "red" or not in_stop_zone:
                self._reset_stop_state(tid)
                self._prev_centroid[tid] = (cx, cy)
                continue

            # ────────────────────────────────────────────────────────
            # From here: signal IS red  AND  vehicle IS in the zone
            # ────────────────────────────────────────────────────────

            vehicle_is_still = (lk_mag < self.LK_STILL_THRESH)

            # ── Gate 1: accumulate still frames ───────────────────────
            if vehicle_is_still:
                self._still_count[tid] += 1
            else:
                # Slowly drain counter — one moving frame doesn't erase
                # several seconds of confirmed stillness
                self._still_count[tid] = max(0, self._still_count[tid] - 2)

            # ── Gate 2: promote to confirmed-stopped ──────────────────
            if (self._still_count[tid] >= self.MIN_STILL_FRAMES
                    and not self._confirmed_stopped[tid]):
                self._confirmed_stopped[tid] = True
                self._stopped_centroid[tid]  = (cx, cy)   # lock rest position

            # ── Gate 3: check for jump (only if confirmed stopped) ────
            if (self._confirmed_stopped[tid]
                    and not self._violation_logged[tid]
                    and self._stopped_centroid[tid] is not None):

                sc           = self._stopped_centroid[tid]
                displacement = np.hypot(cx - sc[0], cy - sc[1])

                if displacement >= self.JUMP_PX:
                    violations.append({"bbox": bbox, "tid": tid})
                    self._violation_logged[tid] = True
                    self._cooldown[tid]         = self.VIOLATION_COOLDOWN

            self._prev_centroid[tid] = (cx, cy)

        # ── 6. Carry forward grayscale frame ──────────────────────────
        self.prev_gray = curr_gray

        return violations, light_color, stop_line_y, buffer_line_y, {
            "vehicles":    vehicles,
            "red_lights":  red_lights,
            "green_lights": green_lights,
        }

    def reset(self):
        self._still_count.clear()
        self._confirmed_stopped.clear()
        self._stopped_centroid.clear()
        self._violation_logged.clear()
        self._cooldown.clear()
        self._prev_centroid.clear()
        self._last_seen.clear()
        self._red_streak   = 0
        self._green_streak = 0
        self._signal_state = "unknown"
        self.prev_gray = None

    def _expire_stale(self, current_time: float) -> None:
        """
        Reap any vehicle track not seen for TRACK_STALE_SEC — previously these
        6 per-vehicle dicts had no cleanup at all (only a full reset() cleared
        them), so every new track ID from a long real-time session accumulated
        forever. Measured: _still_count alone grew from 10 to 39 entries over
        just 150 frames (~5s) of test footage with no plateau.
        """
        stale = [tid for tid, ts in self._last_seen.items()
                 if current_time - ts > self.TRACK_STALE_SEC]
        for tid in stale:
            self._still_count.pop(tid, None)
            self._confirmed_stopped.pop(tid, None)
            self._stopped_centroid.pop(tid, None)
            self._violation_logged.pop(tid, None)
            self._cooldown.pop(tid, None)
            self._prev_centroid.pop(tid, None)
            self._last_seen.pop(tid, None)
