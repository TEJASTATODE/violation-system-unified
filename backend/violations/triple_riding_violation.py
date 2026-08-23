from collections import defaultdict
from time import time

# ── Confidence / geometry ───────────────────────────────────────────────────
TRIPLE_CONF     = 0.25
TRIPLE_CLASS_ID = 0   # model's only class: MORE_THAN_TWO_PERSONS

# ── Confirmation logic (mirrors helmet_violation.py) ────────────────────────
CONFIRM_FRAMES   = 3
CONFIRM_MAJORITY = 2

# ── TTL ───────────────────────────────────────────────────────────────────
LOCK_TTL = 5.0


class TripleRidingViolationDetector:
    """
    Turns raw per-frame triple-riding detections into confirmed violations.

    Single-frame detections are noisy (a partial occlusion or a passing
    pedestrian can trigger a false hit). A track must show the violation
    on CONFIRM_MAJORITY of the last CONFIRM_FRAMES detection calls before
    it's confirmed. Once confirmed, the violation stays "locked" and is
    re-reported every call until the track has been absent for LOCK_TTL
    seconds, so evidence is saved once per real event rather than once
    per detection call.
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

    def process_frame(self, triple_dets: list, current_time: float | None = None) -> list:
        if current_time is None:
            current_time = time()

        self._expire_stale(current_time)

        violations = []

        for det in triple_dets:
            if det["class"] != TRIPLE_CLASS_ID or det["conf"] < TRIPLE_CONF:
                continue

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
        """
        Reap any key not seen for lock_ttl seconds — from _locked AND _pending.
        Previously _pending only got cleaned up once a track reached a locked
        decision; a track seen briefly but never confirmed left a permanent
        entry. Keyed off _last_seen so this covers every track, confirmed or not.
        """
        stale = [k for k, ts in self._last_seen.items() if now - ts > self.lock_ttl]
        for k in stale:
            self._locked.pop(k, None)
            self._pending.pop(k, None)
            self._last_seen.pop(k, None)
