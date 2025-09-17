import cv2
import numpy as np
import time
from datetime import datetime
import pytz
from typing import Optional, Tuple

class CameraService:
    """Handle camera operations and frame processing"""
    
    def __init__(self, camera_source: str = "rtsp://192.168.1.56:8554/stream"):
        self.camera_source = camera_source
        self.cap = None
        self.is_connected = False
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.fps = 0
        
    def connect(self) -> bool:
        """Connect to camera"""
        try:
            if self.cap is not None:
                self.cap.release()
            
            self.cap = cv2.VideoCapture(self.camera_source)
            

            # Set camera properties for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 10)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # Reduce buffer for RTSP
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
           # self.cap.set(cv2.CAP_PROP_FORMAT, -1)  # Auto format
            
            # Test if camera is working
            ret, frame = self.cap.read()
            if ret and frame is not None:
                self.is_connected = True
                print(f"RTSP Camera {self.camera_source} connected successfully")
                return True
            else:
                self.is_connected = False
                return False
                
        except Exception as e:
            print(f"RTSP Camera connection error: {e}")
            self.is_connected = False
            return False
    
    def disconnect(self):
        """Disconnect from camera"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.is_connected = False
        print(f"RTSP Camera {self.camera_source} disconnected")
    
    def read_frame(self) -> Optional[np.ndarray]:
        """Read frame from camera"""
        if not self.is_connected or self.cap is None:
            return None
        
        try:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                self.frame_count += 1
                self._update_fps()
                return frame
            else:
                return None
        except Exception as e:
            print(f"Frame read error: {e}")
            return None
    
    def _update_fps(self):
        """Calculate and update FPS"""
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:  # Update every second
            self.fps = self.frame_count / (current_time - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = current_time
    
    def add_overlays(self, frame: np.ndarray) -> np.ndarray:
        """Add timestamp, FPS and other overlays to frame"""
        if frame is None:
            return frame
        
        try:
            # Get current time in Bucharest timezone
            bucharest_tz = pytz.timezone('Europe/Bucharest')
            current_datetime = datetime.now(bucharest_tz)
            
            date_str = current_datetime.strftime("%d-%m-%Y")
            time_str = current_datetime.strftime("%H:%M:%S")
            
            # Add overlays
            cv2.putText(frame, f"FPS: {self.fps:.1f}", (20, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Date: {date_str}", (20, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Time: {time_str}", (20, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            return frame
            
        except Exception as e:
            print(f"Overlay error: {e}")
            return frame
    
    def resize_frame(self, frame: np.ndarray, width: int = 640, height: int = 480) -> np.ndarray:
        """Resize frame to specified dimensions"""
        if frame is None:
            return frame
        
        try:
            return cv2.resize(frame, (width, height))
        except Exception as e:
            print(f"Frame resize error: {e}")
            return frame
    
    def convert_to_rgb(self, frame: np.ndarray) -> np.ndarray:
        """Convert BGR frame to RGB for Streamlit display"""
        if frame is None:
            return frame
        
        try:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Color conversion error: {e}")
            return frame
    
    def save_frame(self, frame: np.ndarray, filepath: str) -> bool:
        """Save frame to file"""
        if frame is None:
            return False
        
        try:
            return cv2.imwrite(filepath, frame)
        except Exception as e:
            print(f"Frame save error: {e}")
            return False
    
    def get_camera_info(self) -> dict:
        """Get camera information"""
        info = {
            'source': self.camera_source,
            'connected': self.is_connected,
            'fps': self.fps,
            'frame_count': self.frame_count
        }
        
        if self.cap is not None and self.is_connected:
            try:
                info.update({
                    'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    'backend': self.cap.getBackendName()
                })
            except Exception as e:
                print(f"Camera info error: {e}")
        
        return info
    
    @staticmethod
    def get_available_cameras() -> list:
        """Get list of available camera indices"""
        available_cameras = []
        
        for i in range(10):  # Check first 10 indices
            try:
                cap = cv2.VideoCapture(i)
                ret, frame = cap.read()
                if ret and frame is not None:
                    available_cameras.append(i)
                cap.release()
            except:
                continue
        
        return available_cameras

