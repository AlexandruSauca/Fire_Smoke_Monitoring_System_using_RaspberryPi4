import smtplib
import os
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv

import smtplib
import os
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv
import sys

# Load environment variables first
load_dotenv()

print(f"🐍 Python version: {sys.version}")

# Email MIME classes - Python 3.13.5 compatibility check
EMAIL_AVAILABLE = False
MIMEText = None
MIMEMultipart = None
MIMEImage = None

# Try to find the correct email classes for Python 3.13.5
print("🔍 Checking email module availability...")

try:
    # Check what's actually available in email.mime.text
    import email.mime.text as email_text_module
    available_classes = [attr for attr in dir(email_text_module) if not attr.startswith('_')]
    print(f"📧 Available in email.mime.text: {available_classes}")
    
    # Try the most likely names for Python 3.13.5
    if hasattr(email_text_module, 'MIMEText'):
        from email.mime.text import MIMEText
        print("✅ Found MIMEText")
    elif hasattr(email_text_module, 'MimeText'):
        from email.mime.text import MimeText as MIMEText
        print("✅ Found MimeText (using as MIMEText)")
    else:
        raise ImportError("No text MIME class found")
    
    # Check multipart
    import email.mime.multipart as email_multipart_module
    if hasattr(email_multipart_module, 'MIMEMultipart'):
        from email.mime.multipart import MIMEMultipart
        print("✅ Found MIMEMultipart")
    elif hasattr(email_multipart_module, 'MimeMultipart'):
        from email.mime.multipart import MimeMultipart as MIMEMultipart
        print("✅ Found MimeMultipart (using as MIMEMultipart)")
    else:
        raise ImportError("No multipart MIME class found")
    
    # Check image
    import email.mime.image as email_image_module
    if hasattr(email_image_module, 'MIMEImage'):
        from email.mime.image import MIMEImage
        print("✅ Found MIMEImage")
    elif hasattr(email_image_module, 'MimeImage'):
        from email.mime.image import MimeImage as MIMEImage
        print("✅ Found MimeImage (using as MIMEImage)")
    else:
        raise ImportError("No image MIME class found")
    
    EMAIL_AVAILABLE = True
    print("🎉 All email classes loaded successfully!")
    
except Exception as e:
    print(f"❌ Email import failed: {e}")
    print("📧 Email features will be disabled - but app will still work!")
    EMAIL_AVAILABLE = False
    
    # Create safe dummy classes
    class SafeDummy:
        def __init__(self, *args, **kwargs):
            print(f"⚠️ Dummy email class called - email not available")
        def attach(self, *args, **kwargs):
            pass
        def __setitem__(self, key, value):
            pass
        def __getitem__(self, key):
            return ""
    
    MIMEText = SafeDummy
    MIMEMultipart = SafeDummy
    MIMEImage = SafeDummy

class EmailService:
    """Handle email alerts and notifications using your Gmail configuration"""
    
    def __init__(self):
        # Check if email functionality is available
        self.email_available = EMAIL_AVAILABLE
        
        # Use your exact .env configuration
        self.smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "465"))
        self.username = os.getenv("SMTP_USERNAME", "ion.cristian.marius@gmail.com")
        self.password = os.getenv("SMTP_PASSWORD", "jujm hlsm whxk ujmo")
        self.from_email = os.getenv("EMAIL_FROM", "ion.cristian.marius@gmail.com")
        
        # Validate configuration
        if not self.email_available:
            print("⚠️ Email MIME classes not available - email alerts disabled")
        elif not all([self.username, self.password]):
            print("⚠️ Email credentials not configured properly")
        else:
            print(f"✅ Email service configured: {self.username}")
    
    def send_alert_email(self, to_email: str, alert_type: str, room: str, 
                        confidence: float, image_data: Optional[bytes] = None) -> bool:
        """Send fire/smoke alert email"""
        
        if not self.email_available:
            print("⚠️ Email not available - skipping email alert")
            return False
        
        if not to_email or not self._is_configured():
            print(f"⚠️ Email not configured properly")
            return False
        
        try:
            # Create message using the imported classes
            msg = MIMEMultipart()
            msg['From'] = self.from_email
            msg['To'] = to_email
            msg['Subject'] = f"🚨 {alert_type.upper()} ALERT - {room.title()}"
            
            # Email body
            body = self._create_alert_body(alert_type, room, confidence)
            msg.attach(MIMEText(body, 'html'))
            
            # Attach image if provided
            if image_data and self.email_available:
                img = MIMEImage(image_data)
                img.add_header('Content-Disposition', 'attachment', filename=f'{alert_type}_alert.jpg')
                msg.attach(img)
            
            # Send email using SSL (port 465)
            with smtplib.SMTP_SSL(self.smtp_server, self.smtp_port) as server:
                server.login(self.username, self.password)
                server.send_message(msg)
            
            print(f"✅ Alert email sent to {to_email}")
            return True
            
        except Exception as e:
            print(f"❌ Email sending failed: {e}")
            return False
    
    def send_test_email(self, to_email: str) -> bool:
        """Send test email to verify configuration"""
        
        if not self.email_available:
            print("⚠️ Email not available - skipping test email")
            return False
        
        if not to_email or not self._is_configured():
            return False
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.from_email
            msg['To'] = to_email
            msg['Subject'] = "🔥 Fire Monitor - Test Email"
            
            body = """
            <html>
            <body>
                <h2>🔥 Fire & Smoke Monitoring System</h2>
                <p>This is a test email to verify your alert configuration.</p>
                <p><strong>✅ Email alerts are working properly!</strong></p>
                <p>You will receive notifications when fire or smoke is detected.</p>
                <hr>
                <p><small>Sent at: {}</small></p>
            </body>
            </html>
            """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            
            msg.attach(MIMEText(body, 'html'))
            
            with smtplib.SMTP_SSL(self.smtp_server, self.smtp_port) as server:
                server.login(self.username, self.password)
                server.send_message(msg)
            
            print(f"✅ Test email sent to {to_email}")
            return True
            
        except Exception as e:
            print(f"❌ Test email failed: {e}")
            return False
    
    def _create_alert_body(self, alert_type: str, room: str, confidence: float) -> str:
        """Create HTML email body for alerts"""
        
        # Choose color and icon based on alert type
        if alert_type.lower() == 'fire':
            color = '#FF4444'
            icon = '🔥'
            urgency = 'CRITICAL'
        else:  # smoke
            color = '#FF8800'
            icon = '💨'
            urgency = 'WARNING'
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; margin: 0; padding: 20px;">
            <div style="max-width: 600px; margin: 0 auto; border: 3px solid {color}; border-radius: 10px; padding: 20px;">
                <h1 style="color: {color}; text-align: center; margin-top: 0;">
                    {icon} {urgency} ALERT {icon}
                </h1>
                
                <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 20px 0;">
                    <h2 style="color: #333; margin-top: 0;">{alert_type.upper()} DETECTED</h2>
                    <p><strong>Location:</strong> {room.title()}</p>
                    <p><strong>Confidence:</strong> {confidence:.1%}</p>
                    <p><strong>Detected at:</strong> {timestamp}</p>
                </div>
                
                <div style="background-color: {color}; color: white; padding: 15px; border-radius: 5px; text-align: center;">
                    <h3 style="margin: 0;">IMMEDIATE ACTION REQUIRED</h3>
                    <p style="margin: 5px 0;">Please verify the situation and take appropriate safety measures.</p>
                </div>
                
                <div style="margin-top: 20px; padding: 10px; background-color: #e9ecef; border-radius: 5px;">
                    <p style="margin: 0; font-size: 12px; color: #6c757d;">
                        This alert was generated by the Fire & Smoke Monitoring System.<br>
                        If this is a false alarm, please check the system configuration.
                    </p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return body
    
    def _is_configured(self) -> bool:
        """Check if email service is properly configured"""
        return bool(self.username and self.password and self.smtp_server and self.email_available)
    
    def get_configuration_status(self) -> dict:
        """Get email configuration status"""
        return {
            'configured': self._is_configured(),
            'email_available': self.email_available,
            'smtp_server': self.smtp_server,
            'smtp_port': self.smtp_port,
            'username': self.username[:3] + '*' * (len(self.username) - 6) + self.username[-3:] if self.username else '',
            'from_email': self.from_email
        }