import streamlit as st
import cv2
import socket
import json
import time
import numpy as np
from ultralytics import YOLO
import time
from datetime import datetime
import pytz
import os
import hmac
from dotenv import load_dotenv
import pygame
import sys
from pathlib import Path

# Add services to path
sys.path.append(str(Path(__file__).parent))

# Import our services
from services.firebase_config import initialize_firebase, is_firebase_available
from services.alert_service import AlertService
from services.detection_service import DetectionResult
from services.camera_service import CameraService

# ------------------ SERVO CONTROL CONFIGURATION ------------------
RASPBERRY_PI_IP = "192.168.1.56"  
SERVO_CONTROL_PORT = 8899
SERVO_HOLD_TIME = 17

# Load environment and initialize
load_dotenv()
pygame.mixer.init()
initialize_firebase()

# Initialize services
alert_service = AlertService()

# Set page config
st.set_page_config(
    page_title="Fire & Smoke Monitor",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Display logo at the top
st.image("logo-vision.png", width=120) 
# ------------------ SERVO CONTROL FUNCTIONS ------------------
def send_servo_command(command_type, hold_time=SERVO_HOLD_TIME):
    """
    Send command to Raspberry Pi to control servos
    
    Args:
        command_type: 'fire_detected', 'smoke_detected', 'center', or 'status'
        hold_time: seconds to hold position (default: 17)
    
    Returns:
        dict: Response from Raspberry Pi or error message
    """
    try:
        # Create command data
        command_data = {
            "type": command_type,
            "hold_time": hold_time
        }
        
        # Create socket connection
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.settimeout(5.0)  # 5 second timeout
        
        # Connect to Raspberry Pi
        client_socket.connect((RASPBERRY_PI_IP, SERVO_CONTROL_PORT))
        print(f"📡 Connected to Raspberry Pi at {RASPBERRY_PI_IP}:{SERVO_CONTROL_PORT}")
        
        # Send command as JSON
        command_json = json.dumps(command_data)
        client_socket.send(command_json.encode('utf-8'))
        print(f"📤 Sent command: {command_type}")
        
        # Receive response
        response_data = client_socket.recv(1024).decode('utf-8')
        response = json.loads(response_data)
        
        client_socket.close()
        
        print(f"📥 Received response: {response}")
        return response
        
    except socket.timeout:
        error_msg = "Connection to Raspberry Pi timed out"
        print(f"⚠️ {error_msg}")
        return {"status": "error", "message": error_msg}
    except ConnectionRefusedError:
        error_msg = f"Cannot connect to Raspberry Pi at {RASPBERRY_PI_IP}:{SERVO_CONTROL_PORT}"
        print(f"⚠️ {error_msg}")
        return {"status": "error", "message": error_msg}
    except Exception as e:
        error_msg = f"Error sending servo command: {e}"
        print(f"❌ {error_msg}")
        return {"status": "error", "message": error_msg}

def notify_servo_detection(detection_type):
    """
    Notify Raspberry Pi servo system about detection
    This will stop the servos for 17 seconds
    
    Args:
        detection_type: 'fire' or 'smoke'
    """
    command_type = f"{detection_type}_detected"
    response = send_servo_command(command_type, SERVO_HOLD_TIME)
    
    if response.get("status") == "success":
        print(f"✅ Servo system notified: {detection_type} detection - holding for {SERVO_HOLD_TIME}s")
        return True
    else:
        print(f"⚠️ Failed to notify servo system: {response.get('message', 'Unknown error')}")
        return False

def test_servo_connection():
    """Test connection to Raspberry Pi servo system"""
    response = send_servo_command("status")
    if response.get("status") == "success":
        print("✅ Servo system connected and operational")
        print(f"   - Is paused: {response.get('is_paused', False)}")
        print(f"   - Pan position: {response.get('pan_position', 0)}μs")
        print(f"   - Scanning active: {response.get('scanning_active', False)}")
        return True
    else:
        print(f"❌ Servo system not reachable: {response.get('message', 'Unknown error')}")
        return False
def notify_servo_detection(detection_type):
    """
    Notify Raspberry Pi servo system about detection
    This will stop the servos for 17 seconds
    
    Args:
        detection_type: 'fire' or 'smoke'
    """
    command_type = f"{detection_type}_detected"
    response = send_servo_command(command_type, SERVO_HOLD_TIME)
    
    if response.get("status") == "success":
        print(f"✅ Servo system notified: {detection_type} detection - holding for {SERVO_HOLD_TIME}s")
        return True
    else:
        print(f"⚠️ Failed to notify servo system: {response.get('message', 'Unknown error')}")
        return False

def test_servo_connection():
    """Test connection to Raspberry Pi servo system"""
    response = send_servo_command("status")
    if response.get("status") == "success":
        print("✅ Servo system connected and operational")
        print(f"   - Is paused: {response.get('is_paused', False)}")
        print(f"   - Pan position: {response.get('pan_position', 0)}μs")
        print(f"   - Scanning active: {response.get('scanning_active', False)}")
        return True
    else:
        print(f"❌ Servo system not reachable: {response.get('message', 'Unknown error')}")
        return False
    
def test_servo_connection():
    """Test connection to Raspberry Pi servo system"""
    response = send_servo_command("status")
    if response.get("status") == "success":
        print("✅ Servo system connected and operational")
        print(f"   - Is paused: {response.get('is_paused', False)}")
        print(f"   - Pan position: {response.get('pan_position', 0)}μs")
        print(f"   - Scanning active: {response.get('scanning_active', False)}")
        return True
    else:
        print(f"❌ Servo system not reachable: {response.get('message', 'Unknown error')}")
        return False

# ------------------ AUDIO FUNCTIONS ------------------
def play_fire_alarm():
    """Play fire alarm sound for 5 seconds"""
    try:
        if Path("alarm_fire.mp3").exists():
            pygame.mixer.music.load("alarm_fire.mp3")
            pygame.mixer.music.set_volume(0.7)
            pygame.mixer.music.play()  # Play once, not loop
            print("Fire alarm started - 5 second duration")
            return True
        elif Path("alarm_fire.wav").exists():
            pygame.mixer.music.load("alarm_fire.wav")
            pygame.mixer.music.set_volume(0.7)
            pygame.mixer.music.play()
            print("Fire alarm started (WAV) - 5 second duration")
            return True
        else:
            print("No fire alarm file found")
            return False
    except Exception as e:
        print(f"Error playing fire alarm: {e}")
        return False

def play_smoke_alarm():
    """Play smoke alarm sound for 5 seconds"""
    import threading
    try:
        if Path("alarm_smoke.mp3").exists():
            pygame.mixer.music.load("alarm_smoke.mp3")
            pygame.mixer.music.set_volume(0.7)
            pygame.mixer.music.play()  
            print("Smoke alarm started")
            # Stop the alarm after 5 seconds
            def stop_after_5_seconds():
                time.sleep(5)
                pygame.mixer.music.stop()
                print("Smoke alarm stopped after 5 seconds")
            
            threading.Thread(target=stop_after_5_seconds, daemon=True).start()
            return True
        elif Path("alarm_smoke.wav").exists():
            pygame.mixer.music.load("alarm_smoke.wav")
            pygame.mixer.music.set_volume(0.7)
            pygame.mixer.music.play()
            print("Smoke alarm started (WAV) - 5 second duration")
            # Stop the alarm after 5 seconds
            def stop_after_5_seconds():
                time.sleep(5)
                pygame.mixer.music.stop()
                print("Smoke alarm stopped after 5 seconds")
            
            threading.Thread(target=stop_after_5_seconds, daemon=True).start()
            return True
        else:
            # Fallback to fire alarm
            print("No smoke alarm file, using fire alarm")
            return play_fire_alarm()
    except Exception as e:
        print(f"Error playing smoke alarm: {e}")
        return False

def stop_alarm():
    """Stop all alarms"""
    try:
        pygame.mixer.music.stop()
        print("Alarm stopped")
    except Exception as e:
        print(f"Error stopping alarm: {e}")


def get_audio_status():
    """Get audio system status"""
    audio_files = []
    if Path("alarm_fire.mp3").exists():
        audio_files.append("alarm_fire.mp3")
    if Path("alarm_fire.wav").exists():
        audio_files.append("alarm_fire.wav")
    if Path("alarm_smoke.mp3").exists():
        audio_files.append("alarm_smoke.mp3")
    if Path("alarm_smoke.wav").exists():
        audio_files.append("alarm_smoke.wav")
    
    return {
        'initialized': pygame.mixer.get_init() is not None,
        'audio_files_found': len(audio_files),
        'available_files': audio_files
    }

# ------------------ SESSION STATE INIT ------------------
if "password_correct" not in st.session_state:
    st.session_state["password_correct"] = False
if "stop_monitoring" not in st.session_state:
    st.session_state.stop_monitoring = False
if "camera_index" not in st.session_state:
    st.session_state.camera_index = -1  # Start with -1 to force initialization
if "current_room" not in st.session_state:
    st.session_state.current_room = None  # Track the current room
if "user_email" not in st.session_state:
    st.session_state.user_email = None
if "detection_count" not in st.session_state:
    st.session_state.detection_count = 0
if "last_alert_time" not in st.session_state:
    st.session_state.last_alert_time = 0
if "current_alert" not in st.session_state:
    st.session_state.current_alert = None
# Add camera pool for faster switching
if "camera_pool" not in st.session_state:
    st.session_state.camera_pool = {}
if "show_room_history" not in st.session_state:
    st.session_state.show_room_history = False
if "selected_history_room" not in st.session_state:
    st.session_state.selected_history_room = None
if "sidebar_room_selection" not in st.session_state:
    st.session_state.sidebar_room_selection = "Select Room"

# ------------------ PASSWORD CHECK ------------------
def check_password():
    """Check if the password is correct."""
    if st.session_state.get("password_correct", False):
        return True

    # Remove header color, keep header simple
    st.markdown("""
    <h1 style='text-align: center; margin-bottom: 40px;'>
        Fire and Smoke Monitoring System
    </h1>
    """, unsafe_allow_html=True)
    
    # Create centered login form
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("Login"):
            # Email field (longer width)
            email = st.text_input(
                "Email",
                placeholder="your-email@example.com (optional)",
                key="login_email"
            )
            # Password field (longer width)
            password = st.text_input(
                "Password",
                type="password",
                placeholder="Enter your access password",
                key="login_password"
            )
            # Submit button
            submitted = st.form_submit_button("Login", use_container_width=True, type="primary")

            # Emergency number in smaller text
            st.markdown("""
            <div style='background-color: #FF0000; padding: 10px; border-radius: 8px; margin: 15px 0;'>
                <span style='display: block; text-align: center; color: white; font-size: 16px; font-weight: bold;'>EMERGENCY: 112</span>
            </div>
            """, unsafe_allow_html=True)
            # Fire safety tips
            st.info("""
            **Fire Safety Tips:**
            1. **Stay Low:** Crawl under smoke to avoid toxic fumes
            2. **Exit Immediately:** Don't gather belongings - evacuate now
            3. **Check Doors:** Feel door handles - if hot, find another exit
            """)
            
        if submitted:
            if hmac.compare_digest(password, os.getenv("Streamlit_pwd", "12345")):
                st.session_state["password_correct"] = True
                if email:
                    st.session_state["user_email"] = email
                    # Update email in Firebase
                    try:
                        from services.auth_service import AuthService
                        auth_service = AuthService()
                        auth_service.ensure_user_document_exists()
                        auth_service.update_subscriber_email(email)
                    except Exception as e:
                        print(f"Error updating email: {e}")
                st.rerun()
            else:
                st.error("Incorrect password. Access denied.")

    return

# Check password before proceeding
if not check_password():
    st.stop()

# ------------------ SIDEBAR AFTER LOGIN ------------------
with st.sidebar:
    st.markdown("## 🔥 Fire & Smoke Monitor")
    st.markdown("---")
    
    # Account section
    with st.expander("👤 Account", expanded=False):
        if st.session_state.get("user_email"):
            st.info(f"📧 {st.session_state['user_email']}")
        else:
            st.warning("No email configured")
        
        # Logout button
        if st.button("Logout", use_container_width=True):
            st.session_state["password_correct"] = False
            st.session_state["user_email"] = None
            st.rerun()
    
    # Room History section with dropdown
    st.markdown("### 📊 Room History")
    
    # Room selector dropdown (same style as camera selector)
    room_options = ["Select Room", "Warehouse", "Office"]
    selected_room_history = st.selectbox(
        "Select Room for History",
        room_options,
        index=room_options.index(st.session_state.sidebar_room_selection),
        label_visibility="collapsed"
    )
    # Update session state
    st.session_state.sidebar_room_selection = selected_room_history
    
    # Show history button (only if room selected)
    if selected_room_history != "Select Room":
        if st.button(f"View {selected_room_history} History", use_container_width=True):
            st.session_state.show_room_history = True
            st.session_state.selected_history_room = selected_room_history.lower()

# ------------------ UTILITY FUNCTIONS ------------------
def show_quick_stats():
    """Show quick statistics"""
    try:
        stats = alert_service.get_total_stats()
        room_stats = alert_service.get_room_stats()
        
        st.markdown("### 📊 Quick Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Alerts", stats.get('total_alerts', 0))
        with col2:
            st.metric("Fire Alerts", stats.get('total_fire', 0))
        with col3:
            st.metric("Smoke Alerts", stats.get('total_smoke', 0))
        with col4:
            st.metric("Rooms", len(room_stats))
        
        if room_stats:
            st.markdown("**Room Activity:**")
            for room, data in room_stats.items():
                st.write(f"• {room.title()}: {data.get('total_alerts', 0)} alerts")
    
    except Exception as e:
        st.error(f"Error loading statistics: {e}")

def detect_available_cameras():
    """Detect which cameras are available on the system"""
    available_cameras = []
    
    print("🔍 Detecting available cameras...")
    
    # Test camera indices 0-4 (covers most setups)
    for i in range(5):
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    available_cameras.append(i)
                    # Get camera info
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    print(f"📷 Camera {i}: Available ({width}x{height})")
                else:
                    print(f"📷 Camera {i}: Opens but no frame")
            else:
                print(f"📷 Camera {i}: Not available")
            cap.release()
        except Exception as e:
            print(f"📷 Camera {i}: Error - {e}")
    
    print(f"✅ Found {len(available_cameras)} working cameras: {available_cameras}")
    return available_cameras

def setup_camera(camera_name):
    """Setup camera based on selection - using RTSP for warehouse, USB for office"""
    
    # Determine camera source and room based on camera name
    if "Warehouse" in camera_name:
        camera_source = "rtsp://192.168.1.56:8554/stream"
        room_name = "warehouse"
        camera_key = "warehouse_rtsp"
        print(f"Setting up Warehouse RTSP Camera for {room_name} room")
    elif "Office" in camera_name:
        camera_source = 0  # USB camera index
        room_name = "office"
        camera_key = "office_usb"
        print(f"Setting up Office USB Camera for {room_name} room")
    else:
        camera_source = "rtsp://192.168.1.56:8554/stream"
        room_name = "warehouse"
        camera_key = "warehouse_rtsp"
    
    # Update the current room in session state
    st.session_state.current_room = room_name
    
    # Check if we need to switch cameras
    current_camera_key = st.session_state.get("current_camera_key", None)
     # Always disconnect and reconnect when switching cameras, especially for warehouse RTSP
    if current_camera_key != camera_key:
        print(f"Switching from {current_camera_key} to {camera_key}")
        
        # Always release current camera service if exists
        if "camera_service" in st.session_state and st.session_state.camera_service:
            try:
                st.session_state.camera_service.disconnect()
                print(f"Disconnected from previous camera: {current_camera_key}")
            except Exception as e:
                print(f"Warning: Error disconnecting previous camera: {e}")
            
            # Clear the camera service from session state
            st.session_state.camera_service = None
        
        # Create new camera service with fresh connection
        try:
            print(f"Creating new camera service for {camera_key} with source: {camera_source}") 
            st.session_state.camera_service = CameraService(camera_source)             
            # Connect to camera
            if st.session_state.camera_service.connect():
                st.session_state.current_camera_key = camera_key
                print(f"Successfully connected to {camera_key}")
                 # For warehouse RTSP, verify we're actually getting frames
                if camera_key == "warehouse_rtsp":
                    test_frame = st.session_state.camera_service.read_frame()
                    if test_frame is not None:
                        print(f"✅ RTSP stream verified - receiving frames from server")
                    else:
                        print(f"⚠️ RTSP connected but no frames received")
                
                return st.session_state.camera_service
            else:
                print(f"Failed to connect to {camera_key}")
                st.session_state.camera_service = None
                return None
                
        except Exception as e:
            print(f"Error setting up camera service for {camera_key}: {e}")
            st.session_state.camera_service = None
            return None
    
    # Return existing camera service if already connected to the right camera
    return st.session_state.get("camera_service", None)  # For same camera selection, verify connection is still active
    # For same camera selection, verify connection is still active
    existing_service = st.session_state.get("camera_service", None)
    if existing_service and existing_service.is_connected:
        # For warehouse RTSP, always verify we're still getting frames
        if camera_key == "warehouse_rtsp":
            test_frame = existing_service.read_frame()
            if test_frame is not None:
                print(f"✅ {camera_key} connection verified - still receiving frames")
                return existing_service
            else:
                print(f"⚠️ {camera_key} connection lost - reconnecting...")
                # Force reconnection
                st.session_state.current_camera_key = None
                return setup_camera(camera_name)
        else:
            return existing_service
    else:
        print(f"⚠️ Camera service not connected - reconnecting to {camera_key}")
        # Force reconnection
        st.session_state.current_camera_key = None
        return setup_camera(camera_name)

def get_room_from_camera_index(camera_index):
    """Get room name from camera index"""
    if camera_index == 0:
        return "warehouse"
    elif camera_index == 1:
        return "office"
    else:
        return "unknown"
def show_room_history_page(room_name):
    """Display history for a specific room"""
    st.markdown(f"## 📊 {room_name.title()} Room History")
    st.markdown("---")
    
    try:
        # Get all alerts and filter by room
        all_alerts = alert_service.get_recent_alerts(limit=100)
        
        # Debug: Show all room names in alerts to identify the issue
        print(f"Debug: Looking for room '{room_name}'")
        print(f"Debug: Available rooms in alerts: {set(alert.get('room', 'NO_ROOM') for alert in all_alerts)}")
        
        # Filter by room (case-insensitive to handle any case mismatches)
        room_alerts = [alert for alert in all_alerts if alert.get('room', '').lower() == room_name.lower()]
        
        print(f"Debug: Found {len(room_alerts)} alerts for room '{room_name}'")
        print(f"Debug: Total alerts available: {len(all_alerts)}")
        
        if room_alerts:
            # Statistics for this room
            col1, col2, col3 = st.columns(3)
            
            fire_count = len([a for a in room_alerts if a['type'] == 'fire'])
            smoke_count = len([a for a in room_alerts if a['type'] == 'smoke'])
            
            with col1:
                st.metric("Total Alerts", len(room_alerts))
            with col2:
                st.metric("🔥 Fire Alerts", fire_count)
            with col3:
                st.metric("💨 Smoke Alerts", smoke_count)
            
            st.markdown("---")
            st.markdown("### Alert Details")
            
            # Display alerts in reverse chronological order
            for alert in room_alerts:
                alert_time = datetime.fromisoformat(alert['timestamp'].replace('Z', '+00:00'))
                formatted_date = alert_time.strftime("%Y-%m-%d")
                formatted_time = alert_time.strftime("%H:%M:%S")
                alert_type = alert['type'].upper()
                confidence = alert.get('confidence', 0) * 100
                
                # Color coding
                if alert['type'] == 'fire':
                    st.markdown(f"""
                    <div style="background-color: #ffebee; border-left: 4px solid #f44336; padding: 10px; margin: 10px 0; border-radius: 5px; color: black;">
                        <strong style="color: #f44336;">🔥 {alert_type} ALERT</strong><br>
                        <span style="color: black;">📅 Date: {formatted_date}</span><br>
                        <span style="color: black;">⏰ Time: {formatted_time}</span><br>
                        <span style="color: black;">🎯 Confidence: {confidence:.1f}%</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:  # smoke
                    st.markdown(f"""
                    <div style="background-color: #fffde7; border-left: 4px solid #ff9800; padding: 10px; margin: 10px 0; border-radius: 5px; color: black;">
                        <strong style="color: #ff9800;">💨 {alert_type} ALERT</strong><br>
                        <span style="color: black;">📅 Date: {formatted_date}</span><br>
                        <span style="color: black;">⏰ Time: {formatted_time}</span><br>
                        <span style="color: black;">🎯 Confidence: {confidence:.1f}%</span>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info(f"No alerts recorded for {room_name.title()} room yet.")
        
        # Back button
        if st.button("← Back to Monitoring", use_container_width=True, type="primary"):
            # Complete reset of all session state for clean return
            st.session_state.show_room_history = False
            st.session_state.selected_history_room = None
            st.session_state.sidebar_room_selection = "Select Room"
            st.session_state.current_alert = None
            st.session_state.stop_monitoring = False
            # Reset any other monitoring-related state for completely clean start
            if 'detection_count' in st.session_state:
                st.session_state.detection_count = 0
            if 'last_alert_time' in st.session_state:
                st.session_state.last_alert_time = 0
            # Force a complete page refresh with scroll to top
            st.rerun()
            
    except Exception as e:
        st.error(f"Error loading room history: {e}")

# ------------------ MAIN CONTENT ------------------
# IMPORTANT: Use simple if-else to ensure only ONE interface shows
if st.session_state.show_room_history and st.session_state.selected_history_room:
    # Show ONLY history page
    show_room_history_page(st.session_state.selected_history_room)
    
else:
    # Show ONLY monitoring interface
    # Force scroll to top with CSS
    st.markdown("""
    <style>
    .main > div {
        padding-top: 0rem;
    }
    html {
        scroll-behavior: auto;
    }
    </style>
    <script>
    setTimeout(function() {
        window.scrollTo({top: 0, behavior: 'instant'});
        if (window.parent) {
            window.parent.scrollTo({top: 0, behavior: 'instant'});
        }
    }, 100);
    </script>
    <h1 style='text-align: center; color: #FF4B4B; margin-bottom: 30px;'>
        Fire and Smoke Monitoring System
    </h1>
    """, unsafe_allow_html=True)
    
    # ------------------ PROFESSIONAL LAYOUT ------------------
    # Create three main columns: Left (Statistics + Controls), Center (Camera), Right (Settings)
    left_col, center_col, right_col = st.columns([1, 2, 1])
    
    # LEFT COLUMN - Statistics and Camera Source
    with left_col:
        st.markdown("### Statistics")
        
        # Get and display statistics in 2x2 grid
        try:
            stats = alert_service.get_total_stats()
            room_stats = alert_service.get_room_stats()
            
            # Row 1: Total Alerts and Rooms Monitored
            col1_1, col1_2 = st.columns(2)
            with col1_1:
                st.metric("Total Alerts", stats.get('total_alerts', 0))
            with col1_2:
                st.metric("Rooms Monitored", 2)  # Fixed to 2 as requested
            
            # Row 2: Fire Alerts and Smoke Alerts
            col2_1, col2_2 = st.columns(2)
            with col2_1:
                st.metric("Fire Alerts", stats.get('total_fire', 0))
            with col2_2:
                st.metric("Smoke Alerts", stats.get('total_smoke', 0))
        
        except Exception as e:
            st.error(f"Error loading statistics: {e}")
        
        st.markdown("---")
        
        # Camera Source Selector
        st.markdown("### Camera Source")
        camera_options = [
            "Warehouse Camera", 
            "Office Camera"
        ]
        selected_camera = st.selectbox("Select Camera", camera_options, label_visibility="collapsed")
        
        # Auto-set room based on camera and update session state
        if "Warehouse" in selected_camera:
            selected_room = "warehouse"
        elif "Office" in selected_camera:
            selected_room = "office"

        # Store in session state for persistence
        st.session_state.current_room = selected_room
    
    # CENTER COLUMN - Camera Feed
    with center_col:
        # Room title above camera - centered
        st.markdown(f"<h3 style='text-align: center;'>Room: {selected_room.upper()}</h3>", unsafe_allow_html=True)
        
        # Display current alert if any
        alert_placeholder = st.empty()
        if st.session_state.current_alert:
            alert_type = st.session_state.current_alert
            if alert_type == "fire":
                alert_placeholder.error(f"FIRE DETECTED in {selected_room.upper()}")
            else:
                alert_placeholder.warning(f"SMOKE DETECTED in {selected_room.upper()}")
        
        # Camera feed placeholder
        frame_placeholder = st.empty()
    
    # RIGHT COLUMN - Controls and Settings
    with right_col:
        st.markdown("### Controls")
        
        # Detection Thresholds
        st.markdown("**Detection Thresholds**")
        fire_threshold = st.slider("Fire Detection", 0.1, 1.0, 0.7, 0.05)
        smoke_threshold = st.slider("Smoke Detection", 0.1, 1.0, 0.4, 0.05)
        
        st.markdown("**Audio Settings**")
        alarm_volume = st.slider("Alarm Volume", 0.0, 1.0, 0.7, 0.1)
        pygame.mixer.music.set_volume(alarm_volume)
        
        st.markdown("---")
        
        # Control Buttons
        if st.button("Stop Alarms", use_container_width=True, type="primary"):
            stop_alarm()
            st.session_state.current_alert = None
            st.success("Alarms stopped")
        
        enable_alerts = st.checkbox("Enable Alerts", value=True)
    
    st.markdown("---")
    
    # ------------------ CAMERA SETUP ------------------
    # Setup camera service automatically (this also sets the room)
    camera_service = setup_camera(selected_camera)
    
    # Get the actual room being monitored from session state
    actual_room = st.session_state.current_room
    
    # Load YOLO model
    @st.cache_resource
    def load_model():
        try:
            model = YOLO("quantized_model.onnx")
            print(f"✅ YOLO model loaded successfully")
            return model
        except Exception as e:
            st.error(f" Failed to load YOLO model: {e}")
            return None
    
    model = load_model()
    
    if not model:
        center_col.warning("⚠️ YOLO model not available - monitoring without detection")
    
    # ------------------ MONITORING LOGIC ------------------
    # Auto-start monitoring when camera is selected
    st.session_state.stop_monitoring = False
    
    # Monitoring variables
    frame_count = 0
    inference_results = None
    start_time = time.time()
    last_detection_time = 0
    
    # Camera info based on actual room
    camera_info = f"{actual_room.upper()} CAM"
    
    print(f"🎬 Starting monitoring for room: {actual_room}")
    
    # ------------------ MAIN MONITORING LOOP ------------------
    if camera_service and camera_service.is_connected:
        while camera_service.is_connected and not st.session_state.stop_monitoring:
            frame = camera_service.read_frame()
            if frame is None:
                center_col.error("Camera disconnected")
                time.sleep(1)
                continue
            
            # Get current time and use camera service FPS
            current_datetime = datetime.now(pytz.timezone('Europe/Bucharest'))
            time_str = current_datetime.strftime("%H:%M:%S")
            date_str = current_datetime.strftime("%d-%m-%Y")
            
            # Use camera service FPS instead of calculating our own
            fps = camera_service.fps
            
            # Add overlays using camera service
            frame = camera_service.add_overlays(frame)
            
            # Resize frame using camera service
            frame = camera_service.resize_frame(frame, 640, 480)
            frame_count += 1
        
            # Run inference every 10 frames
            detections_found = []
            if frame_count % 10 == 0 and model:
                try:
                    inference_results = model(frame)
                    
                    # Process results
                    if inference_results:
                        for result in inference_results:
                            if hasattr(result, 'boxes') and result.boxes is not None:
                                for box in result.boxes:
                                    conf = float(box.conf[0])
                                    label = model.names[int(box.cls[0])]
                                    
                                    # Apply confidence thresholds using slider values
                                    if label == "smoke" and conf < smoke_threshold:
                                        continue
                                    if label == "fire" and conf < fire_threshold:
                                        continue
                                    
                                    # Create detection object
                                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                                    detection = DetectionResult(label, conf, (x1, y1, x2, y2))
                                    detections_found.append(detection)
                                    
                                    st.session_state.detection_count += 1
                                    
                                    print(f"Frame {frame_count}: {label} detected in {actual_room} "
                                          f"at ({x1},{y1},{x2},{y2}) conf={conf:.2f}")
                
                except Exception as e:
                    print(f"Inference error: {e}")
        
            # Process detections
            high_priority_alert = False
            if detections_found:
                for detection in detections_found:
                    # Draw detection box
                    x1, y1, x2, y2 = detection.bbox
                    color = (0, 0, 255) if detection.label == "fire" else (255, 255, 0)
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{detection.label} {detection.confidence:.2f}", 
                               (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Check for high priority alerts
                    if detection.label == "fire" and detection.confidence >= 0.7:
                        high_priority_alert = True
                    elif detection.label == "smoke" and detection.confidence >= 0.85:
                        high_priority_alert = True
        
            # Handle alerts - use the actual room from session state
            if high_priority_alert and enable_alerts:
                current_alert_time = time.time()
                
                # Prevent spam alerts (10 second cooldown)
                if current_alert_time - st.session_state.last_alert_time > 10:
                    st.session_state.last_alert_time = current_alert_time
                    
                    # Get the highest confidence detection
                    best_detection = max(detections_found, key=lambda x: x.confidence)
                    
                    # Play alarm for 5 seconds only
                    if best_detection.label == "fire":
                        play_fire_alarm()
                        st.session_state.current_alert = "fire"
                    else:
                        play_smoke_alarm()
                        st.session_state.current_alert = "smoke"
                    
                    # Create alert in database with the correct room
                    try:
                        print(f"📝 Creating alert for room: {actual_room}, type: {best_detection.label}")
                        alert_service.create_alert(
                            room=actual_room,  # Use the actual room from session state
                            detection_type=best_detection.label,
                            frame=frame,
                            detections=[best_detection]
                        )
                        print(f"✅ Alert created in database for room: {actual_room}")
                    except Exception as e:
                        print(f"Error creating alert: {e}")
            
            # Add overlays using camera service (this includes camera info, date, time, FPS)
           # frame = camera_service.add_overlays(frame) # Already called above
            
            # Add inference status on LEFT SIDE
            if frame_count % 10 == 0 and model:
                cv2.putText(frame, "INFERENCE: RUNNING", (10, 150),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)  # Yellow for inference
            else:
                cv2.putText(frame, "INFERENCE: IDLE", (10, 150),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)  # Gray for idle
            
            
            # Convert and display frame using camera service
            frame_rgb = camera_service.convert_to_rgb(frame)
            frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            
            # Small delay to prevent excessive CPU usage
            time.sleep(0.033)  # ~30 FPS
    else:
        center_col.error("❌ Camera service not available or not connected")
        center_col.info("Check RTSP server connection or USB camera")
    
    # Footer
    st.markdown("---")
    st.markdown("*Fire & Smoke Monitoring System*")