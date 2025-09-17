# Alternative using ONNX Runtime (if you have .onnx model)
# pip install onnxruntime

import cv2
import numpy as np
import onnxruntime as ort
import torch

class YOLOv8ONNX:
    def __init__(self, model_path, input_size=640, conf_threshold=0.6, nms_threshold=0.4):
        # Initialize ONNX Runtime session
        self.session = ort.InferenceSession(model_path)
        
        # Get input/output info
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        print(f"Input name: {self.input_name}")
        print(f"Output name: {self.output_name}")
        print(f"Input shape: {self.session.get_inputs()[0].shape}")
        print(f"Output shape: {self.session.get_outputs()[0].shape}")
        
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        
        self.class_names = ['smoke', 'fire']
        self.class_colors = {
            0: (128, 128, 128), # Gray for smoke
            1: (0, 0, 255)      # Red for fire
        }
    
    def preprocess(self, image):
        # Resize and pad image
        h, w = image.shape[:2]
        scale = min(self.input_size / h, self.input_size / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        resized = cv2.resize(image, (new_w, new_h))
        padded = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        padded[:new_h, :new_w] = resized
        
        # Convert to RGB and normalize
        padded = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        input_tensor = padded.astype(np.float32) / 255.0
        input_tensor = np.transpose(input_tensor, (2, 0, 1))  # HWC to CHW
        input_tensor = np.expand_dims(input_tensor, axis=0)  # Add batch dimension
        
        return input_tensor, scale
    
    def detect(self, image):
        # Preprocess
        input_tensor, scale = self.preprocess(image)
        
        # Run inference
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        predictions = outputs[0]
        
        # Post-process
        detections = []
        for i, detection in enumerate(predictions[0].T):  # Transpose for easier access
            scores = detection[4:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > self.conf_threshold:
                # Extract box coordinates
                x_center, y_center, width, height = detection[:4]
                x1 = (x_center - width / 2) / scale
                y1 = (y_center - height / 2) / scale
                x2 = (x_center + width / 2) / scale
                y2 = (y_center + height / 2) / scale
                
                detections.append([x1, y1, x2, y2, confidence, class_id])
        
        # Apply NMS
        if detections:
            detections = np.array(detections)
            boxes = detections[:, :4]
            scores = detections[:, 4]
            
            indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), 
                                     self.conf_threshold, self.nms_threshold)
            
            if len(indices) > 0:
                return detections[indices.flatten()]
        
        return []

def main():
    # Initialize model
    model_path = "quantized_model.onnx"  # Make sure this matches your ONNX model path
    detector = YOLOv8ONNX(model_path)
    
    rtsp_url = "rtsp://192.168.1.56:8554/stream"  # Replace with your RTSP stream URL
    
    # Initialize video capture
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    print("Starting detection... Press 'q' to quit")
    
    while True:
        cap.grab()
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        # Get detections
        detections = detector.detect(frame)
        
        # Draw detections
        for det in detections:
            x1, y1, x2, y2, conf, class_id = map(float, det)
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            class_id = int(class_id)
            
            # Draw bounding box
            color = detector.class_colors[class_id]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Add label with confidence
            label = f"{detector.class_names[class_id]} {conf:.2f}"
            (label_width, label_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            cv2.rectangle(
                frame,
                (x1, y1 - label_height - 10),
                (x1 + label_width, y1),
                color,
                -1,
            )
            cv2.putText(
                frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )
        
        # Display FPS
        cv2.imshow("YOLOv8 ONNX Detection", frame)
        
        # Break on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()