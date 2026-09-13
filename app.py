import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolo11n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam started. Press Q to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read webcam frame.")
        break

    # Object detection + tracking
    results = model.track(
        frame,
        persist=True,
        conf=0.5,
        verbose=False
    )

    # Draw boxes, labels and tracking IDs
    annotated_frame = results[0].plot()

    # Display output
    cv2.imshow("CodeAlpha - Object Detection and Tracking", annotated_frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()