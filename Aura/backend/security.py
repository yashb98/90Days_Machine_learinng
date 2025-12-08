import os
import firebase_admin
from firebase_admin import credentials, auth
from fastapi import WebSocket


def init_firebase():
    try:
        if not firebase_admin._apps:
            # Check Cloud Run path vs Local path
            possible_paths = ["/app/serviceAccountKey.json",
                              "serviceAccountKey.json"]
            key_path = next(
                (p for p in possible_paths if os.path.exists(p)), None)

            if key_path:
                cred = credentials.Certificate(key_path)
                firebase_admin.initialize_app(cred)
                print(f"✅ Firebase Admin Initialized using {key_path}")
            else:
                print(
                    "⚠️ WARNING: serviceAccountKey.json not found. Auth checks will fail.")
    except Exception as e:
        print(f"❌ Firebase Init Error: {e}")


async def verify_token(websocket: WebSocket) -> bool:
    # Skip auth if Firebase failed to load (Dev mode)
    if not firebase_admin._apps:
        return True

    token = websocket.query_params.get("token")
    if not token:
        return False

    try:
        auth.verify_id_token(token)
        return True
    except:
        return False
