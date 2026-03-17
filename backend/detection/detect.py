from detection.yolo_model import detect_general, detect_helmet


def detect_general_objects(frame):
    results    = detect_general(frame)
    detections = []

    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0]

            # Get track_id from ByteTrack
            track_id = int(box.id[0]) if box.id is not None else -1

            detections.append({
                "class"   : int(box.cls[0]),
                "conf"    : float(box.conf[0]),
                "box"     : (int(x1), int(y1), int(x2), int(y2)),
                "track_id": track_id
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