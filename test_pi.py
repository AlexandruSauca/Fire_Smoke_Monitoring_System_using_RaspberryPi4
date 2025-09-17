#!/usr/bin/env python3

import argparse
import cv2
import numpy as np
import onnxruntime as ort
import sys
import time
from datetime import datetime

class YOLOv8ONNX:
    def __init__(self, model_path, conf_thresh=0.4):
        self.conf_thresh = conf_thresh
        providers = ['CPUExecutionProvider']
        
        try:
            self.session = ort.InferenceSession(model_path, providers=providers)
            print(f"[INFO] YOLO model loaded: {model_path}")
        except Exception as e:
            print(f"[ERROR] Failed to load ONNX model: {e}")
            sys.exit(1)

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.img_size = (self.input_shape[3], self.input_shape[2])
        self.names = ["smoke", "fire"]

    def preprocess(self, image):
        img = cv2.resize(image, self.img_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, 0)
        return img

    def infer(self, frame):
        try:
            img = self.preprocess(frame)
            outputs = self.session.run([self.output_name], {self.input_name: img})[0]
            return outputs
        except Exception as e:
            print(f"[ERROR] Inference failed: {e}")
            return None

    def postprocess(self, frame, preds):
        if preds is None:
            return frame, []
        
        h, w = frame.shape[:2]
        detections = []
        detected_objects = []

        for det in preds[0]:
            conf = det[4]
            cls_id = int(det[5])
            if conf > self.conf_thresh and cls_id < len(self.names):
                x, y, bw, bh = det[0:4]
                x1 = int((x - bw / 2) * w / self.img_size[0])
                y1 = int((y - bh / 2) * h / self.img_size[1])
                x2 = int((x + bw / 2) * w / self.img_size[0])
                y2 = int((y + bh / 2) * h / self.img_size[1])
                detections.append((x1, y1, x2, y2, conf, cls_id))
                detected_objects.append(self.names[cls_id])

        for (x1, y1, x2, y2, conf, cls_id) in detections:
            label = f"{self.names[cls_id]} {conf:.2f}"
            color = (0, 0, 255) if self.names[cls_id] == "fire" else (255, 165, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return frame, detected_objects

def main():
    parser = argparse.ArgumentParser(description="Smoke/Fire Detection from RTSP Stream on Windows")
    parser.add_argument("--model", type=str, default="quantized_model.onnx", help="Path to YOLO ONNX model")
    parser.add_argument("--rtsp", type=str, default="rtsp://192.168.1.56:8554/stream", help="RTSP stream URL from Raspberry Pi")
    parser.add_argument("--conf-thresh", type=float, default=0.4, help="Confidence threshold")
    parser.add_argument("--monitor", action="store_true", help="Show video preview")
    
    args = parser.parse_args()
    
    # Initialize YOLO
    yolo = YOLOv8ONNX(args.model, args.conf_thresh)
    
    # Open RTSP stream with robust decoding settings
    cap = cv2.VideoCapture(args.rtsp)
    if not cap.isOpened():
        print(f"[ERROR] Failed to open RTSP stream: {args.rtsp}")
        print("Ensure the Raspberry Pi stream is active, network is reachable, and port 8554 is open.")
        sys.exit(1)
    
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # Smaller buffer to reduce lag
    cap.set(cv2.CAP_PROP_FPS, 20)  # Match Pi's framerate
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))  # Force H.264
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print(f"[INFO] Streaming from: {args.rtsp}")
    
    frame_count = 0
    last_detection = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Failed to grab frame, attempting to reconnect...")
                cap.release()
                time.sleep(2)
                cap = cv2.VideoCapture(args.rtsp)
                if not cap.isOpened():
                    print("[ERROR] Reconnection failed, exiting...")
                    break
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
                cap.set(cv2.CAP_PROP_FPS, 20)
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
                continue
            
            # YOLO inference
            preds = yolo.infer(frame)
            frame, detected = yolo.postprocess(frame, preds)
            
            # Alert on detection
            if detected:
                current_time = time.time()
                if current_time - last_detection > 3:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    print(f"[ALERT] {timestamp} - {', '.join(detected)}")
                    last_detection = current_time
            
            # Add timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Local preview
            if args.monitor:
                cv2.imshow("Smoke/Fire Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_count += 1
            if frame_count % 150 == 0:
                print(f"[INFO] Frames processed: {frame_count}")
                
    except KeyboardInterrupt:
        print("\n[INFO] Stopping...")
    finally:
        cap.release()
        if args.monitor:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()