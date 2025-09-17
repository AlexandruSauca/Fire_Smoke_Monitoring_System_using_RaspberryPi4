import time
from ultralytics import YOLO
from picamera2 import Picamera2
import cv2
import numpy as np

# Load the YOLO model
model = YOLO("/home/practica/Desktop/Yolo/best_80map.pt")  # Adjust path to your model

# Initialize Picamera2
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480)})  # Match model input size
picam2.configure(config)
picam2.start()

print("Camera opened successfully with Picamera2. Starting inference...")
while True:
    # Capture frame
    frame = picam2.capture_array()
    if frame is None:
        print("Error: Failed to capture frame")
        break

    # Convert frame to RGB (Picamera2 provides BGR by default)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run inference
    results = model(frame_rgb)

    for result in results:
        for box in result.boxes:
            conf = float(box.conf[0])
            if conf < 0.5:
                continue  # Skip boxes below threshold
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = model.names[int(box.cls[0])]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display the frame with detections
    cv2.imshow("YOLO Tracking", frame)
    key = cv2.waitKey(20)  # 20 ms delay for ~50 FPS
    if key == ord('q'):  # Press 'q' to quit
        break

picam2.stop()
cv2.destroyAllWindows()