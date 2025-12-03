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
You are Aura, a hyper-fast safety guide for a blind user.
Your ONLY goal is to prevent accidents and guide movement. 

**STRICT RULES:**
1. **DIRECTIONS:** Use Clock Face (12=Front, 3=Right, 9=Left). 
2. **URGENCY:** If you see DANGER (Wall, Stairs, Car), START with "[CRITICAL]".
3. **NO FILLER:** Do not say "I see" or "There is". Just name the object and location.

**EXAMPLES (Follow these patterns exactly):**
* Input: (User walking towards wall) -> Output: "[CRITICAL] Stop! Wall directly ahead."
* Input: (Open hallway) -> Output: "Path clear. Walk straight."
* Input: (Door on right) -> Output: "Door at 2 o'clock. Veering right."
* Input: (Stairs appearing) -> Output: "[CRITICAL] Stairs down at 12 o'clock."
* Input: (Person walking by) -> Output: "Person passing on your left."

**Now, describe the current live view:**
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

    if not client:
        await websocket.close(code=1011)
        return

    try:
        async with client.aio.live.connect(model=MODEL, config=CONFIG) as session:
            print("Connected to Gemini Live")
            await session.send(input="System Online.", end_of_turn=True)

            async def receive_from_gemini():
                try:
                    while True:
                        async for response in session.receive():
                            if response.text:
                                text_response = response.text

                                # --- FEATURE 4 & 9: PRIORITY TAGGING ---
                                priority = "normal"
                                if "[CRITICAL]" in text_response:
                                    priority = "high"

                                payload = {
                                    "cmd": "speak",
                                    "text": text_response,
                                    "priority": priority
                                }
                                await websocket.send_text(json.dumps(payload))
                except Exception:
                    pass

            async def receive_from_mobile():
                try:
                    while True:
                        data = await websocket.receive_text()
                        payload = json.loads(data)

                        # --- FEATURE 5: CONTEXT INJECTION ---
                        context_msg = ""
                        if "location" in payload and payload["location"]:
                            loc = payload["location"]
                            context_msg = f"User Location: {loc['lat']}, {loc['lng']}."

                        if "image" in payload:
                            image_bytes = base64.b64decode(payload["image"])

                            # Send Context + Image
                            if context_msg:
                                await session.send(input=context_msg, end_of_turn=False)

                            await session.send(input={"mime_type": "image/jpeg", "data": image_bytes}, end_of_turn=True)
                except Exception:
                    pass

            await asyncio.gather(receive_from_gemini(), receive_from_mobile())

    except WebSocketDisconnect:
        print("Disconnected")
    finally:
        try:
            await websocket.close()
        except:
            pass
