import os
import firebase_admin
from firebase_admin import credentials, auth
from fastapi import WebSocket


def init_firebase():
    """Initializes Firebase Admin SDK safely."""
    try:
        if not firebase_admin._apps:
            possible_paths = [
                "/app/serviceAccountKey.json",
                "serviceAccountKey.json",
                "/secrets/serviceAccountKey.json"
            ]
            key_path = next(
                (p for p in possible_paths if os.path.exists(p)), None)

            if key_path:
                cred = credentials.Certificate(key_path)
                firebase_admin.initialize_app(cred)
                print(f"Firebase Admin Initialized using {key_path}")
            else:
                print("WARNING: serviceAccountKey.json not found.")
    except Exception as e:
        print(f"Firebase Init Error: {e}")


async def verify_token(websocket: WebSocket) -> bool:
    """Verifies the Firebase ID token from the WebSocket query params."""
    if not firebase_admin._apps:
        # If Firebase isn't set up, open mode (dev)
        return True

    token = websocket.query_params.get("token")
    if not token:
        return False

    try:
        auth.verify_id_token(token)
        return True
    except:
        return False
