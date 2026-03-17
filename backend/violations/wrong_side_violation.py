from collections import defaultdict
import numpy as np
from utils.lane_utils import (
    get_vehicle_zone,
    get_box_area,
    get_box_center,
    MIN_MOVEMENT,
    MIN_DX
)

# ── Constants ─────────────────────────────────────────
CONFIRM_FRAMES  = 5
MIN_AREA        = 3000
HISTORY_FRAMES  = 8
VEHICLE_CLASSES = {2, 3, 5, 7}

# ── Global state ──────────────────────────────────────
vehicle_history   = {}
locked_violations = {}


class VehicleState:
    def __init__(self, track_id):
        self.track_id       = track_id
        self.boxes          = []
        self.areas          = []
        self.centers        = []
        self.wrong_count    = 0
        self.violation      = False
        self.zone           = None

    def update(self, box, zone):
        self.boxes.append(box)
        self.areas.append(get_box_area(box))
        self.centers.append(get_box_center(box))
        self.zone = zone

        if len(self.boxes) > HISTORY_FRAMES:
            self.boxes.pop(0)
            self.areas.pop(0)
            self.centers.pop(0)

    def get_avg_dx(self):
        """
        Average horizontal movement per frame.
        Uses last 5 center points.
        Positive = moving right
        Negative = moving left
        """
        if len(self.centers) < 2:
            return 0.0
        points = self.centers[-5:]
        dxs    = [
            points[i+1][0] - points[i][0]
            for i in range(len(points)-1)
        ]
        return float(np.mean(dxs)) if dxs else 0.0

    def get_avg_dy(self):
        """
        Average vertical movement per frame.
        Positive = moving down = toward camera
        """
        if len(self.centers) < 2:
            return 0.0
        points = self.centers[-5:]
        dys    = [
            points[i+1][1] - points[i][1]
            for i in range(len(points)-1)
        ]
        return float(np.mean(dys)) if dys else 0.0

    def get_total_movement(self):
        """Total movement magnitude per frame."""
        dx = self.get_avg_dx()
        dy = self.get_avg_dy()
        return float(np.sqrt(dx**2 + dy**2))

    def is_confirmed(self):
        return self.wrong_count >= CONFIRM_FRAMES


def is_wrong_side(zone, dx, dy, total_movement):
    """
    Core wrong side logic.

    Rules:
    1. Must be moving (not parked)
    2. LEFT zone  + moving RIGHT = wrong ❌
    3. RIGHT zone + moving LEFT  = wrong ❌
    4. CENTER zone + strong horizontal = wrong ❌
    """
    # Skip parked/slow vehicles
    if total_movement < MIN_MOVEMENT:
        return False

    if zone == "left":
        # Oncoming lane
        # Wrong = moving toward your lane (right)
        return dx > MIN_DX

    elif zone == "right":
        # Your lane
        # Wrong = moving toward oncoming (left)
        return dx < -MIN_DX

    elif zone == "center":
        # Buffer zone
        # Wrong = strong horizontal movement
        return abs(dx) > MIN_DX * 1.5

    return False


def wrong_side_violation(general_detections,
                         frame_width,
                         center_x=None):
    """
    Main violation detection function.

    Args:
        general_detections : from detect_general_objects()
        frame_width        : actual frame width
        center_x           : detected road center x
    """
    global vehicle_history, locked_violations

    violations   = []
    all_vehicles = []

    for det in general_detections:

        if det["class"] not in VEHICLE_CLASSES:
            continue

        track_id = det.get("track_id", -1)
        box      = det["box"]
        conf     = det["conf"]

        if track_id == -1:
            continue

        if get_box_area(box) < MIN_AREA:
            continue

        # Get zone using detected center
        zone = get_vehicle_zone(box, frame_width, center_x)

        # Create or update state
        if track_id not in vehicle_history:
            vehicle_history[track_id] = VehicleState(track_id)

        state = vehicle_history[track_id]
        state.update(box, zone)

        # Already locked
        if track_id in locked_violations:
            violations.append({
                "box"     : box,
                "track_id": track_id,
                "zone"    : zone,
                "conf"    : conf
            })
            all_vehicles.append({
                "box"        : box,
                "track_id"   : track_id,
                "zone"       : zone,
                "violation"  : True,
                "approaching": True
            })
            continue

        # Get movement
        dx             = state.get_avg_dx()
        dy             = state.get_avg_dy()
        total_movement = state.get_total_movement()

        # Check wrong side
        wrong = is_wrong_side(
            zone, dx, dy, total_movement
        )

        if wrong:
            state.wrong_count += 1
        else:
            state.wrong_count = max(
                0, state.wrong_count - 1
            )

        # Confirm
        if state.is_confirmed():
            locked_violations[track_id] = True
            state.violation = True
            print(f"🚨 Wrong side locked: "
                  f"track_{track_id} "
                  f"zone={zone} "
                  f"dx={dx:.1f}")

            violations.append({
                "box"     : box,
                "track_id": track_id,
                "zone"    : zone,
                "conf"    : conf
            })

        approaching = dy > 1 and total_movement > MIN_MOVEMENT

        all_vehicles.append({
            "box"        : box,
            "track_id"   : track_id,
            "zone"       : zone,
            "violation"  : state.violation,
            "approaching": approaching
        })

    return violations, all_vehicles


def reset_wrong_side():
    global vehicle_history, locked_violations
    vehicle_history   = {}
    locked_violations = {}
    print("🔄 Wrong side reset!")