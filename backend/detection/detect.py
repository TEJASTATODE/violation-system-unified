from detection.yolo_model import detect_general, detect_helmet, detect_triple, detect_signal, detect_smoke


def detect_general_objects(frame):

    results = detect_general(frame)

    detections = []

    for r in results:

        if r.boxes is None:
            continue

        for box in r.boxes:

            x1, y1, x2, y2 = box.xyxy[0]

            track_id = (
                int(box.id[0])
                if box.id is not None
                else -1
            )

            detections.append({
                "box": (
                    int(x1),
                    int(y1),
                    int(x2),
                    int(y2),
                ),
                "track_id": track_id,
                "conf": float(box.conf[0]),
                "class": int(box.cls[0]),
            })

    return detections


def detect_helmet_objects(frame):
    results    = detect_helmet(frame)
    detections = []

    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0]

            detections.append({
                "class"   : int(box.cls[0]),
                "conf"    : float(box.conf[0]),
                "box"     : (int(x1), int(y1), int(x2), int(y2)),
                "track_id": -1   # helmet model has no tracker
            })

    return detections


def detect_triple_riding_objects(frame):

    results = detect_triple(frame)

    detections = []

    for r in results:

        if r.boxes is None:
            continue

        for box in r.boxes:

            x1, y1, x2, y2 = box.xyxy[0]

            track_id = (
                int(box.id[0])
                if box.id is not None
                else -1
            )

            detections.append({
                "box": (
                    int(x1),
                    int(y1),
                    int(x2),
                    int(y2),
                ),
                "track_id": track_id,
                "conf": float(box.conf[0]),
                "class": int(box.cls[0]),
            })

    return detections


# Per-class confidence for signal.pt (vehicle=0, red_light=1, green_light=2).
# detect_signal() itself calls YOLO at a low conf=0.15 so faint/distant lights
# aren't dropped before we even see them; this per-class filter then applies
# the real thresholds — vehicles need the standard bar, but light classes are
# small, sparse objects that need a lower bar or they're filtered to nothing.
SIGNAL_VEHICLE_CONF = 0.30
SIGNAL_LIGHT_CONF   = 0.18


def detect_signal_objects(frame):

    results = detect_signal(frame)

    detections = []

    for r in results:

        if r.boxes is None:
            continue

        for box in r.boxes:

            x1, y1, x2, y2 = box.xyxy[0]

            # ported from SignalJumpingDetector.process_frame — drop boxes smaller than 20x20px
            if (x2 - x1) < 20 or (y2 - y1) < 20:
                continue

            cls  = int(box.cls[0])
            conf = float(box.conf[0])

            min_conf = SIGNAL_VEHICLE_CONF if cls == 0 else SIGNAL_LIGHT_CONF
            if conf < min_conf:
                continue

            track_id = (
                int(box.id[0])
                if box.id is not None
                else -1
            )

            detections.append({
                "box": (
                    int(x1),
                    int(y1),
                    int(x2),
                    int(y2),
                ),
                "track_id": track_id,
                "conf": conf,
                "class": cls,
            })

    return detections


def detect_smoke_objects(frame):

    results = detect_smoke(frame)

    detections = []

    for r in results:

        if r.boxes is None:
            continue

        for box in r.boxes:

            x1, y1, x2, y2 = box.xyxy[0]

            track_id = (
                int(box.id[0])
                if box.id is not None
                else -1
            )

            detections.append({
                "box": (
                    int(x1),
                    int(y1),
                    int(x2),
                    int(y2),
                ),
                "track_id": track_id,
                "conf": float(box.conf[0]),
                "class": int(box.cls[0]),
            })

    return detections