import cv2
import numpy as np
from datetime import datetime
from typing import List, Optional
from PIL import Image
import io
import uuid
import os

from .firebase_config import get_firestore_client, get_storage_bucket, is_firebase_available
from .detection_service import DetectionResult
from .email_service import EmailService
from .auth_service import AuthService

# Import firestore for query operations
try:
    from firebase_admin import firestore
except ImportError:
    firestore = None

class AlertService:
    """Handle alert creation, storage and notifications"""
    
    # Define valid rooms
    VALID_ROOMS = ['warehouse', 'office', 'test_room']
    
    def __init__(self):
        self.firebase_available = is_firebase_available()
        
        if self.firebase_available:
            self.db = get_firestore_client()
            self.bucket = get_storage_bucket()
            
            # Collection references
            self.alerts_ref = self.db.collection('alerts')
            self.rooms_ref = self.db.collection('rooms')
            self.users_ref = self.db.collection('users')
            
            # Initialize default rooms
            self._ensure_rooms_exist()
        else:
            print("⚠️ Firebase not available - alerts will not be saved to database")
            self.db = None
            self.bucket = None
        
        # Initialize services
        self.email_service = EmailService()
        self.auth_service = AuthService()
    
    def _ensure_rooms_exist(self):
        """Ensure default room documents exist"""
        if not self.firebase_available:
            return
        
        try:
            for room_name in self.VALID_ROOMS:
                room_doc = self.rooms_ref.document(room_name)
                if not room_doc.get().exists:
                    room_data = {
                        'name': room_name,
                        'smoke_alerts': 0,
                        'fire_alerts': 0,
                        'total_alerts': 0,
                        'last_alert': None,
                        'created_at': datetime.now().isoformat()
                    }
                    room_doc.set(room_data)
                    print(f"✅ Created room document: {room_name}")
        except Exception as e:
            print(f"Error ensuring rooms exist: {e}")
    
    def _validate_room(self, room: str) -> str:
        """Validate and normalize room name"""
        room_normalized = room.lower().strip()
        
        if room_normalized not in self.VALID_ROOMS:
            print(f"⚠️ Invalid room '{room}' - defaulting to 'warehouse'")
            return 'warehouse'
        
        return room_normalized
    
    def create_alert(self, room: str, detection_type: str, frame: np.ndarray, 
                    detections: List[DetectionResult]) -> bool:
        """Create and process a new alert"""
        
        try:
            # Validate and normalize room name
            room = self._validate_room(room)
            
            # Generate unique alert ID
            alert_id = str(uuid.uuid4())
            timestamp = datetime.now().isoformat()
            
            # Get highest confidence detection
            max_confidence = max([d.confidence for d in detections], default=0.0)
            
            print(f"🚨 Creating alert: {detection_type} in {room} (confidence: {max_confidence:.2f})")
            print(f"   Alert ID: {alert_id}")
            print(f"   Room: {room}")
            print(f"   Timestamp: {timestamp}")
            
            # Save image to Firebase Storage (if available)
            image_url = None
            if self.firebase_available:
                image_url = self._save_alert_image(alert_id, frame, detections, room)
            
            # Create alert document (if Firebase available)
            if self.firebase_available:
                alert_data = {
                    'id': alert_id,
                    'room': room,  # Already validated and normalized
                    'type': detection_type.lower(),
                    'timestamp': timestamp,
                    'image_url': image_url,
                    'confidence': max_confidence,
                    'detection_count': len(detections),
                    'processed': True
                }
                
                # Log before saving
                print(f"📝 Saving alert to Firestore: room={alert_data['room']}, type={alert_data['type']}")
                
                self.alerts_ref.document(alert_id).set(alert_data)
                
                # Update room statistics
                self._update_room_stats(room, detection_type.lower(), timestamp)
                
                # Update user alert count
                self.auth_service.increment_alert_count()
            
            # Send email notification
            self._send_email_notification(room, detection_type, detections, frame)
            
            print(f"✅ Alert processed successfully: {detection_type} in {room} (ID: {alert_id})")
            return True
            
        except Exception as e:
            print(f"❌ Error creating alert: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _save_alert_image(self, alert_id: str, frame: np.ndarray, 
                         detections: List[DetectionResult], room: str) -> Optional[str]:
        """Save alert image to Firebase Storage with room folder"""
        
        if not self.firebase_available or not self.bucket:
            return None
        
        try:
            # Draw detections on frame
            annotated_frame = frame.copy()
            for detection in detections:
                x1, y1, x2, y2 = detection.bbox
                color = (0, 0, 255) if detection.label == 'fire' else (255, 255, 0)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
                cv2.putText(annotated_frame, f"{detection.label} {detection.confidence:.2f}", 
                           (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Add room label to image
            cv2.putText(annotated_frame, f"ROOM: {room.upper()}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Convert to JPEG
            _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            
            # Upload to Firebase Storage with room subfolder
            blob_name = f"alerts/{room}/{alert_id}.jpg"
            blob = self.bucket.blob(blob_name)
            blob.upload_from_string(buffer.tobytes(), content_type='image/jpeg')
            
            # Make blob publicly accessible
            blob.make_public()
            
            print(f"📸 Image saved to: {blob_name}")
            return blob.public_url
            
        except Exception as e:
            print(f"Error saving alert image: {e}")
            return None
    
    def _update_room_stats(self, room: str, detection_type: str, timestamp: str):
        """Update room statistics"""
        
        if not self.firebase_available:
            return
        
        try:
            # Validate room before updating
            room = self._validate_room(room)
            
            room_ref = self.rooms_ref.document(room)
            room_doc = room_ref.get()
            
            if room_doc.exists:
                room_data = room_doc.to_dict()
            else:
                # Create room if it doesn't exist
                room_data = {
                    'name': room,
                    'smoke_alerts': 0,
                    'fire_alerts': 0,
                    'total_alerts': 0,
                    'last_alert': None,
                    'created_at': datetime.now().isoformat()
                }
            
            # Update counters
            room_data['total_alerts'] = room_data.get('total_alerts', 0) + 1
            room_data['last_alert'] = timestamp
            
            if detection_type == 'fire':
                room_data['fire_alerts'] = room_data.get('fire_alerts', 0) + 1
            elif detection_type == 'smoke':
                room_data['smoke_alerts'] = room_data.get('smoke_alerts', 0) + 1
            
            room_data['updated_at'] = datetime.now().isoformat()
            
            # Save updated data
            room_ref.set(room_data)
            
            print(f"📊 Updated stats for room '{room}': Total alerts = {room_data['total_alerts']}")
            
        except Exception as e:
            print(f"Error updating room stats: {e}")
    
    def _send_email_notification(self, room: str, detection_type: str, 
                               detections: List[DetectionResult], frame: np.ndarray):
        """Send email notification if subscriber exists"""
        
        try:
            # Get subscriber email
            subscriber_email = self.auth_service.get_subscriber_email()
            
            if not subscriber_email:
                print("No subscriber email configured")
                return
            
            # Get highest confidence detection
            max_confidence = max([d.confidence for d in detections], default=0.0)
            
            # Convert frame to JPEG for email attachment
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            image_data = buffer.tobytes()
            
            # Send email with correct room information
            success = self.email_service.send_alert_email(
                to_email=subscriber_email,
                alert_type=detection_type,
                room=room,  # This is now the validated room
                confidence=max_confidence,
                image_data=image_data
            )
            
            if success:
                print(f"📧 Email alert sent to {subscriber_email} for room {room}")
            else:
                print(f"❌ Failed to send email alert")
                
        except Exception as e:
            print(f"Error sending email notification: {e}")
    
    def get_recent_alerts(self, limit: int = 50, room: Optional[str] = None) -> List[dict]:
        """Get recent alerts from Firestore, optionally filtered by room"""
        
        if not self.firebase_available:
            return []
        
        try:
            # Build query
            alerts_query = self.alerts_ref
            
            # Filter by room if specified
            if room:
                room = self._validate_room(room)
                alerts_query = alerts_query.where('room', '==', room)
            
            # Order by timestamp and limit
            alerts_query = (alerts_query
                          .order_by('timestamp', direction='DESCENDING')
                          .limit(limit))
            
            alerts = []
            for doc in alerts_query.stream():
                alert_data = doc.to_dict()
                alert_data['id'] = doc.id
                alerts.append(alert_data)
            
            if room:
                print(f"📋 Retrieved {len(alerts)} alerts for room '{room}'")
            else:
                print(f"📋 Retrieved {len(alerts)} total alerts")
            
            return alerts
            
        except Exception as e:
            print(f"Error getting recent alerts: {e}")
            return []
    
    def get_room_stats(self) -> dict:
        """Get statistics for all rooms"""
        
        if not self.firebase_available:
            # Return mock data
            return {
                'warehouse': {'name': 'warehouse', 'total_alerts': 0, 'fire_alerts': 0, 'smoke_alerts': 0},
                'office': {'name': 'office', 'total_alerts': 0, 'fire_alerts': 0, 'smoke_alerts': 0}
            }
        
        try:
            stats = {}
            
            for doc in self.rooms_ref.stream():
                room_data = doc.to_dict()
                room_name = doc.id
                
                # Only include valid rooms
                if room_name in self.VALID_ROOMS:
                    stats[room_name] = room_data
            
            return stats
            
        except Exception as e:
            print(f"Error getting room stats: {e}")
            return {}
    
    def get_total_stats(self) -> dict:
        """Get total system statistics"""
        
        try:
            room_stats = self.get_room_stats()
            
            total_stats = {
                'total_alerts': 0,
                'total_fire': 0,
                'total_smoke': 0,
                'rooms_monitored': len(room_stats),
                'last_alert': None
            }
            
            last_alert_times = []
            
            for room_data in room_stats.values():
                total_stats['total_alerts'] += room_data.get('total_alerts', 0)
                total_stats['total_fire'] += room_data.get('fire_alerts', 0)
                total_stats['total_smoke'] += room_data.get('smoke_alerts', 0)
                
                if room_data.get('last_alert'):
                    last_alert_times.append(room_data['last_alert'])
            
            if last_alert_times:
                total_stats['last_alert'] = max(last_alert_times)
            
            return total_stats
            
        except Exception as e:
            print(f"Error getting total stats: {e}")
            return {
                'total_alerts': 0,
                'total_fire': 0,
                'total_smoke': 0,
                'rooms_monitored': 0,
                'last_alert': None
            }
    
    def clear_room_alerts(self, room: str) -> bool:
        """Clear all alerts for a specific room (admin function)"""
        
        if not self.firebase_available:
            return False
        
        try:
            room = self._validate_room(room)
            
            # Get all alerts for this room
            alerts_query = self.alerts_ref.where('room', '==', room)
            
            # Delete each alert
            batch = self.db.batch()
            deleted_count = 0
            
            for doc in alerts_query.stream():
                batch.delete(doc.reference)
                deleted_count += 1
            
            # Commit the batch delete
            batch.commit()
            
            # Reset room statistics
            room_ref = self.rooms_ref.document(room)
            room_ref.update({
                'smoke_alerts': 0,
                'fire_alerts': 0,
                'total_alerts': 0,
                'last_alert': None,
                'updated_at': datetime.now().isoformat()
            })
            
            print(f"🗑️ Cleared {deleted_count} alerts for room '{room}'")
            return True
            
        except Exception as e:
            print(f"Error clearing room alerts: {e}")
            return False
    
    def test_alert_system(self) -> bool:
        """Test the alert system with a mock alert"""
        
        try:
            # Create test frame
            test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(test_frame, "TEST ALERT", (200, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)
            
            # Create test detection
            test_detection = DetectionResult("fire", 0.95, (100, 100, 300, 300))
            
            # Create alert
            return self.create_alert("test_room", "fire", test_frame, [test_detection])
            
        except Exception as e:
            print(f"Error testing alert system: {e}")
            return False