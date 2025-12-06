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
        key_paths = ["/app/serviceAccountKey.json", "serviceAccountKey.json"]
        key_path = next((p for p in key_paths if os.path.exists(p)), None)

        if key_path:
            cred = credentials.Certificate(key_path)
            firebase_admin.initialize_app(cred)
            print(f" Firebase Admin Initialized")
        else:
            print(" WARNING: serviceAccountKey.json not found.")
except Exception as e:
    print(f" Firebase Init Error: {e}")

# --- 2. GEMINI CLIENT ---
client = None
try:
    if API_KEY:
        client = genai.Client(api_key=API_KEY, http_options={
                              "api_version": "v1alpha"})
        print(" Gemini Client Initialized")
    else:
        print(" WARNING: GEMINI_API_KEY is missing.")
except Exception as e:
    print(f" Gemini Client Init Error: {e}")

MODEL = "models/gemini-2.0-flash-exp"

# --- SYSTEM INSTRUCTION (The Brain) ---
SYS_INSTRUCTION = """
You are Aura, a safety-oriented navigation assistant.
**RULES:**

**Obedience:** Follow user voice commands instantly.
"""

CONFIG = {
    "response_modalities": ["TEXT"],  # Changed to TEXT for FlutterTTS
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

# --- VISION LOGIC (The Eyes) ---


def has_scene_changed(prev_frame, curr_frame, threshold=40):
    if prev_frame is None:
        return False
    try:
        prev_g = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_g = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(prev_g, curr_g)
        return np.mean(diff) > threshold
    except Exception:
        return True


@app.get("/")
async def health_check():
    return {"status": "online", "message": "Aura Brain is Listening"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    # ... (Security checks remain the same) ...
    if not await verify_token(websocket):
        await websocket.close(code=4001)
        return
    if not client:
        await websocket.close(code=1011)
        return

    print("✅ Client Connected")

    try:
        async with client.aio.live.connect(model=MODEL, config=CONFIG) as session:
            print("🚀 Connected to Gemini Live")

            # --- CHANGE 1: THE GREETING ---
            # We ask a question to invite the user to speak
            await session.send(input="... Aura Online. I am listening. What should I look for?", end_of_turn=True)

            async def receive_from_gemini():
                try:
                    # Removed redundant 'while True'
                    async for response in session.receive():
                        if response.text:
                            text_response = response.text
                            print(f"🤖 Gemini: {text_response}")
                            payload = {"cmd": "speak", "text": text_response}
                            await websocket.send_text(json.dumps(payload))
                except Exception as e:
                    print(f"❌ Gemini Receive Error: {e}")

            async def receive_from_mobile():
                prev_frame = None
                while True:  # Moved 'while' OUTSIDE the try block
                    try:
                        data = await websocket.receive_text()
                        payload = json.loads(data)

                        # --- 1. HANDLE VOICE COMMANDS ---
                        if "text" in payload:
                            user_command = payload["text"]
                            print(f"🗣️ User Command: {user_command}")
                            await session.send(input=user_command, end_of_turn=True)

                        # --- 2. HANDLE GPS CONTEXT ---
                        context_msg = ""
                        if "location" in payload and payload["location"]:
                            loc = payload["location"]
                            context_msg = f"User Location: {loc['lat']}, {loc['lng']}."

                        # --- 3. HANDLE VIDEO LOGIC ---
                        if "image" in payload:
                            image_bytes = base64.b64decode(payload["image"])
                            # print(f"📸 Received Image: {len(image_bytes)} bytes")

                            if context_msg:
                                await session.send(input=context_msg, end_of_turn=False)

                            # OpenCV Scene Change Check (Wrapped safely)
                            try:
                                np_arr = np.frombuffer(image_bytes, np.uint8)
                                frame = await asyncio.to_thread(cv2.imdecode, np_arr, cv2.IMREAD_COLOR)
                                if frame is not None:
                                    if await asyncio.to_thread(has_scene_changed, prev_frame, frame):
                                        await websocket.send_json({"cmd": "interrupt"})
                                    prev_frame = frame
                            except Exception:
                                pass  # Ignore CV errors, keep streaming

                            await session.send(input={"mime_type": "image/jpeg", "data": image_bytes}, end_of_turn=True)

                    except WebSocketDisconnect:
                        print("📱 Mobile Disconnected")
                        break  # Stop loop only on disconnect
                    except Exception as e:
                        print(f"❌ Frame Error: {e}")
                        continue  # Skip bad frame and continue
    finally:
        try:
            await websocket.close()
        except:
            pass
