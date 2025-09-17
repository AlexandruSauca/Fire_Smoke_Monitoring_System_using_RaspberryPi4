import firebase_admin
from firebase_admin import credentials, firestore, storage
import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def initialize_firebase():
    """Initialize Firebase Admin SDK with your exact configuration"""
    
    # Prevent multiple initializations
    if firebase_admin._apps:
        return
    
    try:
        # Get credentials file from your .env
        credentials_file = os.getenv("FIRESTORE_CREDENTIALS", "serviceAccountKey.json")
        storage_bucket = os.getenv("STORAGE_BUCKET", "secret-canyon-468210-f9.appspot.com")
        
        service_account_path = Path(credentials_file)
        
        if service_account_path.exists():
            cred = credentials.Certificate(str(service_account_path))
            firebase_admin.initialize_app(cred, {
                'storageBucket': storage_bucket
            })
            print(f"🔥 Firebase initialized successfully")
            print(f"📦 Storage bucket: {storage_bucket}")
        else:
            print(f"❌ Firebase service account key not found: {credentials_file}")
            print("📝 Please ensure serviceAccountKey.json is in your project root")
            # Don't stop the app, just disable Firebase features
            
    except Exception as e:
        print(f"❌ Firebase initialization failed: {str(e)}")
        print("⚠️ Firebase features will be disabled")

def get_firestore_client():
    """Get Firestore client instance"""
    try:
        return firestore.client()
    except Exception as e:
        print(f"❌ Firestore client error: {e}")
        return None

def get_storage_bucket():
    """Get Firebase Storage bucket instance"""
    try:
        return storage.bucket()
    except Exception as e:
        print(f"❌ Storage bucket error: {e}")
        return None

def is_firebase_available() -> bool:
    """Check if Firebase is properly initialized"""
    try:
        db = get_firestore_client()
        bucket = get_storage_bucket()
        return db is not None and bucket is not None
    except:
        return False

# Initialize on import
initialize_firebase()