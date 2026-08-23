from ultralytics import YOLO

general_model = YOLO("models/yolov8n.pt")
helmet_model  = YOLO("models/helmet_best.pt")
triple_model  = YOLO("models/triple_riding_best.pt")
signal_model  = YOLO("models/signal.pt")
smoke_model   = YOLO("models/smoke_best.pt")


def detect_general(frame):

    results = general_model.track(
        frame,
        conf=0.38,
        imgsz=768,
        tracker="bytetrack.yaml",
        persist=True,
        verbose=False
    )

    return results


def detect_helmet(frame):

    results = helmet_model(
        frame,
        conf=0.30,
        imgsz=768,
        verbose=False
    )

    return results


def detect_triple(frame):

    results = triple_model.track(
        frame,
        conf=0.25,
        imgsz=640,
        tracker="bytetrack.yaml",
        persist=True,
        verbose=False
    )

    return results


def detect_signal(frame):

    # conf kept low here — traffic lights are small, sparse objects that get
    # filtered too aggressively by one blanket threshold. Per-class confidence
    # (vehicle vs red/green light) is applied in detect.detect_signal_objects
    # instead. imgsz raised 640->960 for small-object recall (mirrors why
    # helmet_model already uses 768 instead of 640).
    results = signal_model.track(
        frame,
        conf=0.15,
        imgsz=960,
        persist=True,
        verbose=False
    )

    return results


def detect_smoke(frame):

    # .track() instead of .predict(): smoke_emission_violation.py's confirm
    # gate needs a persistent identity per vehicle across frames. Without a
    # tracker, every detection fell back to a 100px grid-quantized centroid,
    # which breaks down for anything but a near-stationary vehicle — verified
    # empirically: a smoking vehicle moving 60px/frame never confirmed once
    # in 12 frames despite being detected every frame, because it kept
    # crossing grid boundaries before CONFIRM_MAJORITY could accumulate.
    results = smoke_model.track(
        frame,
        conf=0.5,
        tracker="bytetrack.yaml",
        persist=True,
        verbose=False
    )

    return results