import json
import asyncio
import os
import base64
import traceback
import numpy as np
import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from google import genai
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, auth

# Load env vars
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

app = FastAPI()

# --- 1. SAFE FIREBASE INIT ---
try:
    if not firebase_admin._apps:
        key_path = "serviceAccountKey.json"
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

# --- 2. GEMINI CLIENT ---
client = None
try:
    if API_KEY:
        client = genai.Client(api_key=API_KEY, http_options={
                              "api_version": "v1alpha"})
        print("✅ Gemini Client Initialized")
    else:
        print("⚠️ WARNING: GEMINI_API_KEY is missing.")
except Exception as e:
    print(f"❌ Gemini Client Init Error: {e}")

MODEL = "models/gemini-2.0-flash-exp"

SYS_INSTRUCTION = """
You are Aura, an advanced navigation guide for a blind user. 
Analyze the video feed from the user's perspective to identify a safe walking path.

**PRIORITIES (In Order):**
1. **IMMEDIATE HAZARDS:** Warn instantly about steps, drops, traffic, or head-level obstacles.
2. **NAVIGATION COMMANDS:** Tell the user where to walk. Use relative directions (e.g., "Walk straight", "Veer slightly right", "Stop").
3. **OBSTACLE LOCATION:** Mention obstacles relative to the user (e.g., "Pole on your left", "Person approaching from right").

**RULES:**
- Be imperative and direct.
- Max 15 words per response.
- Do NOT describe the scene aesthetically. Focus only on walkability.
- If the path is clear, say "Path clear, proceed."
"""

CONFIG = {
    "response_modalities": ["TEXT"],
    "system_instruction": SYS_INSTRUCTION,
}


async def verify_token(websocket: WebSocket):
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

# --- VISION LOGIC (Restored) ---


def has_scene_changed(prev_frame, curr_frame, threshold=40):
    """Checks if the visual scene has shifted significantly."""
    if prev_frame is None:
        return False
    try:
        prev_g = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_g = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(prev_g, curr_g)
        score = np.mean(diff)
        return score > threshold
    except Exception as e:
        print(f"CV2 Error: {e}")
        return False


@app.get("/")
async def health_check():
    return {"status": "online", "message": "Aura Brain is Listening"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    if not await verify_token(websocket):
        await websocket.close(code=4001)
        return

    print("✅ Client Connected")

    if not client:
        await websocket.close(code=1011)
        return

    # State for Vision Logic
    prev_frame = None

    try:
        async with client.aio.live.connect(model=MODEL, config=CONFIG) as session:
            print("🚀 Connected to Gemini Live")

            # Wake Word
            await session.send(input="System Online.", end_of_turn=True)

            async def receive_from_gemini():
                try:
                    while True:
                        async for response in session.receive():
                            if response.text:
                                text_response = response.text
                                print(f"🤖 Gemini Says: {text_response}")

                                payload = {
                                    "cmd": "speak",
                                    "text": text_response
                                }
                                await websocket.send_text(json.dumps(payload))
                except Exception:
                    pass

            async def receive_from_mobile():
                nonlocal prev_frame
                try:
                    while True:
                        data = await websocket.receive_text()
                        payload = json.loads(data)

                        if "image" in payload:
                            image_bytes = base64.b64decode(payload["image"])

                            # --- CV2 PROCESSING ---
                            try:
                                # Convert bytes to CV2 Frame
                                np_arr = np.frombuffer(image_bytes, np.uint8)
                                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                                if frame is not None:
                                    # Check for Barge-In
                                    if has_scene_changed(prev_frame, frame):
                                        print("⚡ SCENE CHANGED! Interrupting...")
                                        await websocket.send_json({"cmd": "interrupt"})

                                    prev_frame = frame
                            except Exception as cv_err:
                                print(f"CV Error: {cv_err}")
                            # ----------------------

                            # Send frame to Gemini
                            await session.send(input={"mime_type": "image/jpeg", "data": image_bytes}, end_of_turn=True)
                except Exception:
                    pass

            await asyncio.gather(receive_from_gemini(), receive_from_mobile())

    except WebSocketDisconnect:
        print("❌ Disconnected")
    except Exception as e:
        print(f"🔥 Error: {e}")
    finally:
        try:
            await websocket.close()
        except:
            pass
