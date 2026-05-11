from ultralytics import YOLO

general_model = YOLO("yolov8n.pt")
helmet_model  = YOLO("models/helmet_best.pt")


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