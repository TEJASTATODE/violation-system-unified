from collections import defaultdict
from time import time

# ── Confirmation logic (mirrors triple_riding_violation.py) ────────────────
CONFIRM_FRAMES   = 3
CONFIRM_MAJORITY = 2

# ── TTL ───────────────────────────────────────────────────────────────────
LOCK_TTL = 5.0


class SmokeEmissionDetector:
    """
    Turns raw per-frame smoke-model detections into confirmed violations.

    Same confirm-over-N-frames + lock-TTL pattern as TripleRidingViolationDetector
    (single-frame detections are noisy; a track must show the detection on
    CONFIRM_MAJORITY of the last CONFIRM_FRAMES calls before it's confirmed).

    Class-semantics caveat, deliberately not "fixed" by guessing: the model
    (backend/models/smoke_best.pt) has two classes, neither meaningfully
    labelled ('0' and a raw Roboflow export string). Checked which one fires
    on this repo's test footage — both do, inconsistently, with no clear
    pattern distinguishing real exhaust smoke from noise. So both classes are
    treated as candidate detections here (no class filter), matching the
    original Smoke_Emission/main.py's behaviour of drawing every returned box.
    TODO: once evidence crops are saved (backend/evidence/smoke_emission/),
    visually inspect them to determine which class index is real smoke and
    add a class filter here.

    detect_smoke() now uses .track() (like every other detector in the
    pipeline except helmet, which doesn't need its own tracker — see
    yolo_model.py), so track_id is normally valid and used as the identity
    key. Grid-quantized centroid remains only as the fallback for a frame
    where tracking momentarily drops a box (matches triple-riding's same
    fallback). Before this, smoke_model used .predict() with no tracker at
    all, forcing every detection through the grid fallback — verified
    empirically that this made the confirm gate never fire for a moving
    vehicle (it kept crossing 100px grid boundaries before CONFIRM_MAJORITY
    could accumulate in any one cell).
    """

    def __init__(self,
                 lock_ttl: float = LOCK_TTL,
                 confirm_frames: int = CONFIRM_FRAMES,
                 confirm_majority: int = CONFIRM_MAJORITY):
        self.lock_ttl         = lock_ttl
        self.confirm_frames   = confirm_frames
        self.confirm_majority = confirm_majority

        self._locked: dict    = {}                    # key -> {"ts": float}
        self._pending: dict   = defaultdict(list)      # key -> [bool, ...]
        self._last_seen: dict = {}                     # key -> ts, for reaping never-confirmed tracks

    def process_frame(self, smoke_dets: list, current_time: float | None = None) -> list:
        if current_time is None:
            current_time = time()

        self._expire_stale(current_time)

        violations = []

        for det in smoke_dets:
            key = self._key(det["box"], det.get("track_id", -1))
            self._last_seen[key] = current_time

            if key in self._locked:
                self._locked[key]["ts"] = current_time
                violations.append({
                    "box":      det["box"],
                    "track_id": det.get("track_id", -1),
                    "conf":     det["conf"],
                })
                continue

            buf = self._pending[key]
            buf.append(True)
            if len(buf) > self.confirm_frames:
                buf.pop(0)

            if len(buf) >= self.confirm_frames and buf.count(True) >= self.confirm_majority:
                self._locked[key] = {"ts": current_time}
                self._pending.pop(key, None)
                violations.append({
                    "box":      det["box"],
                    "track_id": det.get("track_id", -1),
                    "conf":     det["conf"],
                })

        return violations

    def reset(self):
        self._locked.clear()
        self._pending.clear()
        self._last_seen.clear()

    @staticmethod
    def _key(box, track_id, grid=100):
        if track_id != -1:
            return f"track_{track_id}"
        cx = (box[0] + box[2]) // 2
        cy = (box[1] + box[3]) // 2
        return f"grid_{(cx // grid) * grid}_{(cy // grid) * grid}"

    def _expire_stale(self, now):
        """Reap any key not seen for lock_ttl seconds — from _locked AND
        _pending (see triple_riding_violation.py's identical fix for why)."""
        stale = [k for k, ts in self._last_seen.items() if now - ts > self.lock_ttl]
        for k in stale:
            self._locked.pop(k, None)
            self._pending.pop(k, None)
            self._last_seen.pop(k, None)
