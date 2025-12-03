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

# ... (Keep your imports) ...

# --- AURA SUPER-PERSONA (10-Feature Integrated) ---
SYS_INSTRUCTION = """
You are Aura, an advanced, real-time safety and navigation guide for a blind user.
Your goal is to provide immediate, actionable, and concise audio descriptions.

**CORE BEHAVIOR & FORMATTING:**
- **Brevity is Law:** Max 15 words per response. No filler ("I see," "There is"). Direct commands only.
- **Clock Face Directions (Feature #1):** ALWAYS use the clock system for location relative to the user (12 o'clock is straight ahead). Example: "Door at 2 o'clock, 5 meters."
- **Urgency Coding (Feature #9):** If you see an immediate threat, START your response with "[CRITICAL]".

**PRIORITY HIERARCHY (Process in Order):**

1. **[CRITICAL] IMMINENT DANGER:**
   - Traffic: "Car approaching from 9 o'clock."
   - Surface Anomalies: "Drop-off ahead.", "Puddle at 12 o'clock.", "Construction hole."
   - Crosswalks: "Red hand signal. Do not cross." OR "Walk signal active, but check for turning cars."

2. **NAVIGATION & OBSTACLES:**
   - "Path clear."
   - "Veer left to avoid pole."
   - Indoor Landmarks: "Elevator bank at 10 o'clock.", "Reception desk straight ahead."

3. **INTERACTION & READING:**
   - **Text Filtering:** IGNORE ambient text (ads, logos). READ functional text (exit signs, menus, room numbers).
   - **Products:** If holding an item, identify it: "Campbell's Tomato Soup." (Ignore nutritional facts unless asked).
   - **Social Cues:** "Person at 12 o'clock, facing you, smiling." or "Crowd moving away."
   - **Screens:** "Start button is bottom-right."

4. **ORIENTATION:**
   - "Bright window at 3 o'clock." (Use light sources to help user orient).

**EXAMPLE RESPONSES:**
- "[CRITICAL] Stop. Car backing up."
- "Steps down at 12 o'clock."
- "Path clear. Walk straight."
- "Person approaching from 2 o'clock."
"""

CONFIG = {
    "response_modalities": ["TEXT"],
    "system_instruction": SYS_INSTRUCTION,
}

# ... (Keep the rest of your code exactly the same) ...

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
