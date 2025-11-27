import json
import asyncio
import os
import base64
import traceback
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
from google import genai
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, auth

# Load env vars
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

app = FastAPI()

# --- 1. SAFE FIREBASE INIT ---
# We wrap global initialization in a try/except to prevent Cloud Run crashes
try:
    if not firebase_admin._apps:
        # Check multiple possible paths for the key file
        key_path = "serviceAccountKey.json"
        # Cloud Run often mounts secrets at the root or /app/
        if os.path.exists("/app/serviceAccountKey.json"):
            key_path = "/app/serviceAccountKey.json"

        if os.path.exists(key_path):
            cred = credentials.Certificate(key_path)
            firebase_admin.initialize_app(cred)
            print(f"✅ Firebase Admin Initialized using {key_path}")
        else:
            print("⚠️ WARNING: serviceAccountKey.json not found. Auth checks will fail.")
except Exception as e:
    print(f"❌ Firebase Init Error: {e}")

# --- 2. SAFE GEMINI INIT ---
# If this fails, the 'client' variable remains None, but the server stays alive.
client = None
try:
    if API_KEY:
        client = genai.Client(api_key=API_KEY, http_options={
                              "api_version": "v1alpha"})
        print("✅ Gemini Client Initialized")
    else:
        print("⚠️ WARNING: GEMINI_API_KEY is missing from environment.")
except Exception as e:
    print(f"❌ Gemini Client Init Error: {e}")

MODEL = "models/gemini-2.0-flash-exp"
CONFIG = {"response_modalities": ["AUDIO"]}


async def verify_token(websocket: WebSocket):
    # If Firebase failed to load, we skip auth so the app still works for testing
    if not firebase_admin._apps:
        print("🔒 Auth skipped (Firebase not initialized)")
        return True

    token = websocket.query_params.get("token")
    if not token:
        return False
    try:
        decoded_token = auth.verify_id_token(token)
        return True
    except:
        return False


@app.get("/")
async def health_check():
    # This endpoint MUST respond with 200 OK for Cloud Run's health check
    return {"status": "online", "message": "Aura Brain is Listening"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("✅ Client Connected (Mobile)")

    if not await verify_token(websocket):
        await websocket.close(code=4001)
        print("⛔ Security Alert: Invalid Token")
        return

    # Check if the Gemini client failed to initialize at startup
    if not client:
        print("❌ Gemini Client is not ready. Shutting down connection.")
        await websocket.close(code=1011)  # 1011 = Server error
        return

    try:
        async with client.aio.live.connect(model=MODEL, config=CONFIG) as session:
            print("🚀 Connected to Gemini Live")

            # Send initial greeting
            await session.send(input="Say: System Online", end_of_turn=True)

            async def receive_from_gemini():
                try:
                    while True:
                        async for response in session.receive():
                            if response.data:
                                # Correct usage: base64.b64encode directly (no os.)
                                payload = {"audio": base64.b64encode(
                                    response.data).decode('utf-8')}
                                await websocket.send_text(json.dumps(payload))
                except Exception as e:
                    print(f"⚠️ Gemini Receive Error: {e}")

            async def receive_from_mobile():
                try:
                    while True:
                        data = await websocket.receive_text()
                        if not data:
                            continue

                        try:
                            payload = json.loads(data)
                            if "image" in payload:
                                # Correct usage: base64.b64decode directly (no os.)
                                image_bytes = base64.b64decode(
                                    payload["image"])
                                await session.send(input={"mime_type": "image/jpeg", "data": image_bytes}, end_of_turn=False)
                        except Exception as inner_e:
                            print(f"⚠️ Frame Error: {inner_e}")
                            continue
                except WebSocketDisconnect:
                    print("📱 Mobile Disconnected")
                    raise
                except Exception as e:
                    print(f"❌ Mobile Receive Error: {e}")

            await asyncio.gather(receive_from_gemini(), receive_from_mobile())

    except WebSocketDisconnect:
        print("❌ Client Disconnected cleanly")
    except Exception as e:
        print(f"🔥 Session Error: {e}")
        traceback.print_exc()
    finally:
        try:
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.close()
        except:
            pass
