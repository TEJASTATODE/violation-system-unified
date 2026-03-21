import cv2

from violations.wrong_side_violation import process_frame
from utils.lane_utils import draw_roi

cap = cv2.VideoCapture(r"C:\Users\TEJAS\OneDrive\Desktop\Miniproject\test17.mp4")
# cap = cv2.VideoCapture(1)

# FIX Bug 3: check the file actually opened before entering the loop.
# Gives a clear error instead of a silent immediate exit.
if not cap.isOpened():
    raise RuntimeError("Could not open video — check the file path.")

prev_frame = None

while True:

    ret, frame = cap.read()

    if not ret:
        break

    if prev_frame is None:
        prev_frame = frame.copy()
        continue

    # FIX Bug 2: save a copy of the RAW frame BEFORE process_frame()
    # draws annotations onto it.  Optical flow on the next iteration
    # must see clean pixel data, not coloured bounding boxes.
    raw_frame = frame.copy()

    frame = process_frame(frame, prev_frame)
    frame = draw_roi(frame)

    cv2.imshow("wrong", frame)

    # Store the clean raw frame for the next iteration
    prev_frame = raw_frame

    if cv2.waitKey(1) == 27:   # ESC to quit
        break

# FIX Bug 1: always release the capture handle and destroy windows,
# whether we exited because the video ended or ESC was pressed.
cap.release()
cv2.destroyAllWindows()