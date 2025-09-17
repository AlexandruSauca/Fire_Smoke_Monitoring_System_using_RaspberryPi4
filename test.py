import cv2
from openvino.runtime import Core
from ultralytics import YOLO
import json
#import paho.mqtt.client as mqtt

#import requests
#OpenVINO model
model = YOLO("best_albumentations_openvino_model/best_albumentations_openvino_model/")




#link rtsp cu port default
rtsp_url = "rtsp://192.168.1.56:8554/stream" 

# MQTT_BROKER = "10.1.58.32"
# MQTT_PORT = 1883
# MQTT_TOPIC = "camera/detections"

#mqtt_client = mqtt.Client(client_id ="YOLOv8_Detector", callback_api_version=mqtt.CallbackAPIVersion.VERSION1)
#mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)


# Open stream
cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) 


if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# While true, citeste frame
while True:

    for _ in range(2):
        cap.grab()

    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break
    
    frame = cv2.resize(frame, (640, 480))
    # Convert frame to RGB - YOLOv8 are nev RGB format
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # YOLOv8 inference pe frame
    results = model.predict(img, conf=0.4, task='detect')
    bottle_detected = False
    # bounding boxes
    for result in results:
        for box in result.boxes.data:
            x1, y1, x2, y2, conf, cls = box.tolist()
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            label = f"{model.names[int(cls)]}: {conf:.2f}"

            #daca este sticla
            if model.names[int(cls)] == 'bottle' and conf > 0.4:
                bottle_detected = True
                detection_data = {
                    "object": "bottle",
                    "confidence": round(conf, 2),
                    "camera": "1"
                }
                #mqtt_client.publish(MQTT_TOPIC, json.dumps(detection_data))
                #print(f"MQTT Sent: {detection_data}")

            # patrulater + label pe el
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # imageshow dupa ce am adaugat inference + bb
    cv2.imshow("YOLOv8 OpenVINO - IP Camera", frame)

    # 'q' pt exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# close cap, inchidem stream
cap.release()
cv2.destroyAllWindows()
# mqtt_client.disconnect()