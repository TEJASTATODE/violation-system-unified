from ultralytics import YOLO
model = YOLO("models/helmet_best.pt")
print(model.names)