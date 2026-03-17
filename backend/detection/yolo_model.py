from ultralytics import YOLO

general_model = YOLO("yolo11n.pt")
helmet_model  = YOLO("models/helmet_best.pt")


def detect_general(frame):
    # Use .track() not () for tracking
    # Use "bytetrack.yaml" not "bytetrack"
    results = general_model.track(
        frame,
        conf    = 0.30,
        imgsz   = 640,
        tracker = "bytetrack.yaml",  # ← correct format
        persist = True,              # ← maintain IDs across frames
        verbose = False
    )
    return results


def detect_helmet(frame):
    # Helmet model — NO tracking needed
    # Just detect on current frame
    results = helmet_model(
        frame,
        conf    = 0.30,
        imgsz   = 640,
        verbose = False
    )
    return results