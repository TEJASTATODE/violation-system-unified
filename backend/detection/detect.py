import cv2
from yolo_model import detect_objects


video_path = r"C:\Users\TEJAS\OneDrive\Desktop\Miniproject\test.mp4"

cap = cv2.VideoCapture(video_path)

frame_count = 0
skip_frames = 3   

while True:
    ret, frame = cap.read()

    if not ret:
        break


    frame_count += 1


    if frame_count % skip_frames != 0:
        continue


    results = detect_objects(frame)


    for r in results:

        boxes = r.boxes

        if boxes is None:
            continue

        for box in boxes:

            x1, y1, x2, y2 = box.xyxy[0]

            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            conf = float(box.conf[0])
            cls = int(box.cls[0])

            label = f"{cls} {conf:.2f}"

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )


    cv2.imshow("Video Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()