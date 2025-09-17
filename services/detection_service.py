import numpy as np
import random
import cv2
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import time

class DetectionResult:
    """Detection result data structure - matches YOLO output format"""
    
    def __init__(self, label: str, confidence: float, bbox: Tuple[int, int, int, int]):
        self.label = label
        self.confidence = confidence
        self.bbox = bbox  # (x1, y1, x2, y2)
    
    def __repr__(self):
        return f"DetectionResult(label='{self.label}', conf={self.confidence:.2f}, bbox={self.bbox})"

class DetectionService:
    """Handle ML detection inference - adapted for your YOLO model"""
    
    def __init__(self, use_mock: bool = False, model_path: str = "best_80map.pt"):
        self.use_mock = use_mock
        self.model = None
        self.model_path = model_path
        self.class_names = {0: 'fire', 1: 'smoke'}  # Update based on your model
        
        # Mock detection settings
        self.mock_detection_rate = 0.05  # 5% chance of detection in mock mode
        self.last_detection_time = 0
        self.detection_cooldown = 15  # Seconds between mock detections
        
        # Confidence thresholds (matching your original code)
        self.fire_threshold = 0.5
        self.smoke_threshold = 0.8
        
        if not use_mock:
            self._load_real_model()
    
    def _load_real_model(self):
        """Load real YOLO model from best_80map.pt"""
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            print(f"✅ YOLO model loaded successfully: {self.model_path}")
            
            # Get class names from model
            if hasattr(self.model, 'names'):
                self.class_names = self.model.names
                print(f"📋 Model classes: {self.class_names}")
            
        except Exception as e:
            print(f"⚠️ Could not load YOLO model: {e}")
            print("🔄 Falling back to mock detection")
            self.use_mock = True
    
    def detect(self, frame: np.ndarray) -> List[DetectionResult]:
        """Perform detection on frame - matches your original inference logic"""
        if frame is None:
            return []
        
        if self.use_mock:
            return self._mock_detect(frame)
        else:
            return self._real_detect(frame)
    
    def _real_detect(self, frame: np.ndarray) -> List[DetectionResult]:
        """Real YOLO detection - adapted from your original code"""
        if self.model is None:
            return []
        
        try:
            # Run inference (matches your original model(frame) call)
            results = self.model(frame)
            detections = []
            
            # Process results (matches your original loop structure)
            for result in results:
                if hasattr(result, 'boxes') and result.boxes is not None:
                    for box in result.boxes:
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        label = self.model.names[cls_id]
                        
                        # Apply confidence thresholds (exactly like your original code)
                        if label == "smoke" and conf < self.smoke_threshold:
                            continue
                        if label == "fire" and conf < self.fire_threshold:
                            continue
                        
                        # Extract bounding box coordinates
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # Create detection result
                        detections.append(DetectionResult(label, conf, (x1, y1, x2, y2)))
                        
                        # Debug print (matches your original print statement)
                        print(f"Detection: {label} detected at (x1={x1}, y1={y1}, x2={x2}, y2={y2}) conf={conf:.2f}")
            
            return detections
            
        except Exception as e:
            print(f"Detection error: {e}")
            return []
    
    def _mock_detect(self, frame: np.ndarray) -> List[DetectionResult]:
        """Mock detection for testing when model not available"""
        current_time = time.time()
        
        # Respect cooldown period
        if current_time - self.last_detection_time < self.detection_cooldown:
            return []
        
        detections = []
        
        # Randomly generate detections
        if random.random() < self.mock_detection_rate:
            self.last_detection_time = current_time
            
            # Generate random detection
            detection_type = random.choice(['fire', 'smoke'])
            confidence = random.uniform(0.6, 0.95)
            
            # Random bounding box
            h, w = frame.shape[:2]
            x1 = random.randint(50, w//2)
            y1 = random.randint(50, h//2)
            x2 = random.randint(x1 + 80, min(x1 + 200, w-50))
            y2 = random.randint(y1 + 80, min(y1 + 200, h-50))
            
            detections.append(DetectionResult(detection_type, confidence, (x1, y1, x2, y2)))
            print(f"🔥 MOCK DETECTION: {detection_type} (conf: {confidence:.2f})")
        
        return detections
    
    def draw_detections(self, frame: np.ndarray, detections: List[DetectionResult]) -> np.ndarray:
        """Draw detection boxes and labels on frame - matches your original drawing code"""
        if frame is None or not detections:
            return frame
        
        try:
            for detection in detections:
                x1, y1, x2, y2 = detection.bbox
                label = detection.label
                confidence = detection.confidence
                
                # Draw bounding box (matches your original rectangle color)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                # Draw label with confidence (matches your original text)
                label_text = f"{label} {confidence:.2f}"
                cv2.putText(frame, label_text, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Add warning overlay for high-confidence detections
                if label == 'fire' and confidence > 0.8:
                    cv2.putText(frame, "⚠️ FIRE ALERT ⚠️", (20, 160),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                elif label == 'smoke' and confidence > 0.85:
                    cv2.putText(frame, "⚠️ SMOKE ALERT ⚠️", (20, 160),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
            
            return frame
            
        except Exception as e:
            print(f"Detection drawing error: {e}")
            return frame
    
    def filter_detections(self, detections: List[DetectionResult]) -> List[DetectionResult]:
        """Filter detections based on confidence thresholds"""
        filtered = []
        
        for detection in detections:
            # Apply the same thresholds as your original code
            if detection.label == 'smoke' and detection.confidence >= self.smoke_threshold:
                filtered.append(detection)
            elif detection.label == 'fire' and detection.confidence >= self.fire_threshold:
                filtered.append(detection)
        
        return filtered
    
    def has_high_priority_detection(self, detections: List[DetectionResult]) -> bool:
        """Check if any detection requires immediate alert"""
        for detection in detections:
            # High priority: fire >= 0.7 or smoke >= 0.85
            if detection.label == 'fire' and detection.confidence >= 0.7:
                return True
            if detection.label == 'smoke' and detection.confidence >= 0.85:
                return True
        return False
    
    def get_detection_summary(self, detections: List[DetectionResult]) -> Dict:
        """Get summary of detections"""
        summary = {
            'total_detections': len(detections),
            'fire_count': 0,
            'smoke_count': 0,
            'max_confidence': 0,
            'highest_priority': None,
            'alert_worthy': False
        }
        
        for detection in detections:
            if detection.label == 'fire':
                summary['fire_count'] += 1
            elif detection.label == 'smoke':
                summary['smoke_count'] += 1
            
            if detection.confidence > summary['max_confidence']:
                summary['max_confidence'] = detection.confidence
                summary['highest_priority'] = detection
        
        # Check if any detection is alert-worthy
        summary['alert_worthy'] = self.has_high_priority_detection(detections)
        
        return summary
    
    def set_mock_detection_rate(self, rate: float):
        """Set mock detection rate (0.0 to 1.0)"""
        self.mock_detection_rate = max(0.0, min(1.0, rate))
        print(f"🎯 Mock detection rate: {self.mock_detection_rate:.1%}")
    
    def set_thresholds(self, fire_threshold: float, smoke_threshold: float):
        """Update confidence thresholds"""
        self.fire_threshold = fire_threshold
        self.smoke_threshold = smoke_threshold
        print(f"🎯 Thresholds updated - Fire: {fire_threshold}, Smoke: {smoke_threshold}")
    
    def force_mock_detection(self, detection_type: str = 'fire') -> List[DetectionResult]:
        """Force a mock detection for testing"""
        if detection_type not in ['fire', 'smoke']:
            detection_type = 'fire'
        
        confidence = random.uniform(0.8, 0.95)
        bbox = (150, 150, 350, 350)  # Fixed bbox for testing
        
        detection = DetectionResult(detection_type, confidence, bbox)
        print(f"🧪 Forced {detection_type} detection: {confidence:.2f}")
        
        return [detection]
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model"""
        info = {
            'model_loaded': self.model is not None,
            'model_path': self.model_path,
            'use_mock': self.use_mock,
            'class_names': self.class_names,
            'fire_threshold': self.fire_threshold,
            'smoke_threshold': self.smoke_threshold
        }
        
        if self.model and hasattr(self.model, 'device'):
            info['device'] = str(self.model.device)
        
        return info