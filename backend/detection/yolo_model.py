from ultralytics import YOLO
model = YOLO("yolov8n.pt")
def detect_objects(image):

    results = model(image)

    return results