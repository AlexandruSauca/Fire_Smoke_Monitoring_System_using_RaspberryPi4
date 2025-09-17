import bcrypt
import os
from datetime import datetime
from dotenv import load_dotenv
from .firebase_config import get_firestore_client, is_firebase_available

# Load environment variables
load_dotenv()

class AuthService:
    """Handle authentication and user management"""
    
    def __init__(self):
        self.db = get_firestore_client()
        self.firebase_available = is_firebase_available()
        self.users_ref = self.db.collection('users') if self.db else None
        self.default_password = os.getenv("Streamlit_pwd", "12345")
        
        if not self.firebase_available:
            print("⚠️ Firebase not available - authentication will work with default password only")
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    def verify_password(self, password: str) -> bool:
        """Verify password against stored hash or default password"""
        try:
            # Always check against default password first (matches your original code)
            if password == self.default_password:
                if self.firebase_available:
                    self.ensure_user_document_exists()
                return True
            
            # If Firebase available, also check stored hash
            if self.firebase_available and self.users_ref:
                users = list(self.users_ref.limit(1).stream())
                
                if users:
                    user_doc = users[0]
                    user_data = user_doc.to_dict()
                    stored_hash = user_data.get('password_hash', '')
                    
                    if stored_hash:
                        return bcrypt.checkpw(password.encode('utf-8'), stored_hash.encode('utf-8'))
            
            return False
            
        except Exception as e:
            print(f"Password verification error: {e}")
            # Fallback to default password
            return password == self.default_password
    
    def ensure_user_document_exists(self):
        """Ensure user document exists in Firestore"""
        if not self.firebase_available:
            return
        
        try:
            users = list(self.users_ref.limit(1).stream())
            if not users:
                self.create_default_user(self.default_password)
        except Exception as e:
            print(f"Error ensuring user document: {e}")
    
    def create_default_user(self, password: str) -> bool:
        """Create default user on first run"""
        if not self.firebase_available:
            return False
        
        try:
            hashed_password = self.hash_password(password)
            
            user_data = {
                'password_hash': hashed_password,
                'subscriber': None,
                'active': True,
                'subscribed_at': datetime.now().isoformat(),
                'alert_count': 0,
                'created_at': datetime.now().isoformat()
            }
            
            self.users_ref.add(user_data)
            print("✅ Default user created successfully")
            return True
            
        except Exception as e:
            print(f"Error creating default user: {e}")
            return False
    
    def update_password_hash(self, user_ref, password: str):
        """Update user with hashed password"""
        if not self.firebase_available:
            return
        
        try:
            hashed_password = self.hash_password(password)
            user_ref.update({
                'password_hash': hashed_password,
                'updated_at': datetime.now().isoformat()
            })
            print("Password hash updated")
        except Exception as e:
            print(f"Error updating password hash: {e}")
    
    def update_subscriber_email(self, email: str):
        """Update subscriber email for the user"""
        if not self.firebase_available:
            print("⚠️ Firebase not available - email not saved")
            return
        
        try:
            users = list(self.users_ref.limit(1).stream())
            if users:
                user_ref = users[0].reference
                user_ref.update({
                    'subscriber': email,
                    'subscribed_at': datetime.now().isoformat()
                })
                print(f"✅ Subscriber email updated: {email}")
            else:
                # Create user if doesn't exist
                self.create_default_user(self.default_password)
                self.update_subscriber_email(email)
        except Exception as e:
            print(f"Error updating subscriber email: {e}")
    
    def get_subscriber_email(self) -> str:
        """Get current subscriber email"""
        if not self.firebase_available:
            return None
        
        try:
            users = list(self.users_ref.limit(1).stream())
            if users:
                user_data = users[0].to_dict()
                return user_data.get('subscriber')
        except Exception as e:
            print(f"Error getting subscriber email: {e}")
        return None
    
    def increment_alert_count(self):
        """Increment user's alert count"""
        if not self.firebase_available:
            return
        
        try:
            users = list(self.users_ref.limit(1).stream())
            if users:
                user_ref = users[0].reference
                user_data = users[0].to_dict()
                current_count = user_data.get('alert_count', 0)
                user_ref.update({'alert_count': current_count + 1})
        except Exception as e:
            print(f"Error incrementing alert count: {e}")
    
    def get_user_stats(self) -> dict:
        """Get user statistics"""
        default_stats = {
            'subscriber': None,
            'alert_count': 0,
            'subscribed_at': None,
            'active': True
        }
        
        if not self.firebase_available:
            return default_stats
        
        try:
            users = list(self.users_ref.limit(1).stream())
            if users:
                user_data = users[0].to_dict()
                return {
                    'subscriber': user_data.get('subscriber'),
                    'alert_count': user_data.get('alert_count', 0),
                    'subscribed_at': user_data.get('subscribed_at'),
                    'active': user_data.get('active', True)
                }
        except Exception as e:
            print(f"Error getting user stats: {e}")
        
        return default_stats