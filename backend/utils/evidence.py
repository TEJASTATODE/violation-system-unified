import cv2
import os
from datetime import datetime

EVIDENCE_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "evidence"
)

VIOLATION_TYPES = ("wrong_side", "helmet", "triple_riding", "signal_jump", "smoke_emission")


def ensure_evidence_dirs(root: str = EVIDENCE_ROOT) -> None:
    for vt in VIOLATION_TYPES:
        os.makedirs(os.path.join(root, vt), exist_ok=True)


def make_key(box, track_id: int = -1, grid: int = 100) -> str:
    if track_id != -1:
        return f"track_{track_id}"
    cx = (box[0] + box[2]) // 2
    cy = (box[1] + box[3]) // 2
    return f"grid_{(cx // grid) * grid}_{(cy // grid) * grid}"


class EvidenceRecorder:
    """
    One instance per violation type. Saves a cropped, padded evidence
    image the first time a track is seen, then dedups on that same
    track/grid key until `expiry_frames` frames pass without it — mirrors
    the saved_riders/save_crop pattern from the original test_helmet.py.
    """

    def __init__(self, violation_type: str, root: str = EVIDENCE_ROOT,
                 expiry_frames: int = 150, pad: int = 15):
        assert violation_type in VIOLATION_TYPES, f"unknown violation_type: {violation_type}"
        self.violation_type = violation_type
        self.dir            = os.path.join(root, violation_type)
        os.makedirs(self.dir, exist_ok=True)
        self.expiry_frames  = expiry_frames
        self.pad            = pad
        self._saved: dict   = {}   # key -> frame_count last saved

    def expire(self, frame_count: int) -> None:
        for k in list(self._saved.keys()):
            if frame_count - self._saved[k] > self.expiry_frames:
                del self._saved[k]

    def reset(self) -> None:
        self._saved.clear()

    def maybe_save(self, frame, box, frame_count: int, track_id: int = -1,
                    extra_box=None):
        """extra_box: e.g. nearest bike box for helmet, unioned into the crop."""
        key = make_key(box, track_id)
        if key in self._saved:
            return None

        path = self._save_crop(frame, box, extra_box)
        if path:
            self._saved[key] = frame_count
        return path

    def _save_crop(self, frame, box, extra_box):
        h, w = frame.shape[:2]

        if extra_box:
            x1 = min(box[0], extra_box[0]); y1 = min(box[1], extra_box[1])
            x2 = max(box[2], extra_box[2]); y2 = max(box[3], extra_box[3])
        else:
            x1, y1, x2, y2 = box

        x1 = max(0, x1 - self.pad); y1 = max(0, y1 - self.pad)
        x2 = min(w, x2 + self.pad); y2 = min(h, y2 + self.pad)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        ts   = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = os.path.join(self.dir, f"{self.violation_type}_{ts}.jpg")
        cv2.imwrite(path, crop)
        print(f"[EVIDENCE] {path}")
        return path
