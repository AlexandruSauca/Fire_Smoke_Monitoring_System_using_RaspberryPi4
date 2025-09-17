"""
Fire & Smoke Monitoring System - Services Package

This package contains all backend services for the monitoring system:
- firebase_config: Firebase initialization and configuration
- auth_service: User authentication and management  
- detection_service: ML detection and inference using YOLO
- alert_service: Alert creation and management
- email_service: Email notifications and alerts

Adapted for your specific configuration:
- Uses your .env credentials
- Integrates with your YOLO model (best_80map.pt)
- Compatible with your Firebase setup
- Audio handled directly in main app with pygame
"""

__version__ = "1.0.0"
__author__ = "Fire Monitor Team"

# Import main classes for easy access
from .firebase_config import initialize_firebase, get_firestore_client, get_storage_bucket, is_firebase_available
from .auth_service import AuthService
from .detection_service import DetectionService, DetectionResult
from .alert_service import AlertService
from .email_service import EmailService

__all__ = [
    'initialize_firebase',
    'get_firestore_client', 
    'get_storage_bucket',
    'is_firebase_available',
    'AuthService',
    'DetectionService',
    'DetectionResult', 
    'AlertService',
    'EmailService'
]