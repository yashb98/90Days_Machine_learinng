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
1. **Clock Face:** Use "12 o'clock" (front), "3 o'clock" (right).
2. **Urgency:** If danger (cars, drops), start with "[CRITICAL]".
3. **Brevity:** Max 15 words. No filler.
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

    # 1. Security
    if not await verify_token(websocket):
        print("⛔ Invalid Token")
        await websocket.close(code=4001)
        return

    print("✅ Client Connected")

    if not client:
        await websocket.close(code=1011)
        return

    # 2. Connection Loop
    try:
        async with client.aio.live.connect(model=MODEL, config=CONFIG) as session:
            print("🚀 Connected to Gemini Live")

            # FIXED: Add a pause (periods/commas) to prevent "ura" clipping
            await session.send(input="... Aura System Online.", end_of_turn=True)

            # --- Task A: Receive from Gemini (Brain -> Phone) ---
            async def receive_from_gemini():
                try:
                    while True:
                        async for response in session.receive():
                            if response.text:
                                text_response = response.text
                                print(f"🤖 Gemini: {text_response}")

                                payload = {
                                    "cmd": "speak",
                                    "text": text_response
                                }
                                await websocket.send_text(json.dumps(payload))
                except Exception as e:
                    print(f"❌ Gemini Receive Error: {e}")

            # --- Task B: Receive from Mobile (Phone -> Brain) ---
            async def receive_from_mobile():
                prev_frame = None
                try:
                    while True:
                        data = await websocket.receive_text()
                        payload = json.loads(data)

                        if "image" in payload:
                            image_bytes = base64.b64decode(payload["image"])

                            # OpenCV Processing (Run in thread to avoid blocking)
                            try:
                                np_arr = np.frombuffer(image_bytes, np.uint8)
                                frame = await asyncio.to_thread(cv2.imdecode, np_arr, cv2.IMREAD_COLOR)

                                if frame is not None:
                                    # Check Scene Change
                                    if await asyncio.to_thread(has_scene_changed, prev_frame, frame):
                                        print("⚡ Scene Changed -> Interrupting")
                                        await websocket.send_json({"cmd": "interrupt"})

                                    prev_frame = frame
                            except Exception as cv_e:
                                print(f"⚠️ CV Error: {cv_e}")

                            # Send to Gemini
                            await session.send(input={"mime_type": "image/jpeg", "data": image_bytes}, end_of_turn=True)
                except WebSocketDisconnect:
                    print("📱 Mobile Disconnected")
                    raise
                except Exception as e:
                    print(f"❌ Mobile Receive Error: {e}")
                    traceback.print_exc()

            await asyncio.gather(receive_from_gemini(), receive_from_mobile())

    except WebSocketDisconnect:
        print("❌ Disconnected Cleanly")
    except Exception as e:
        print(f"🔥 Critical Server Error: {e}")
        traceback.print_exc()
    finally:
        try:
            await websocket.close()
        except:
            pass
