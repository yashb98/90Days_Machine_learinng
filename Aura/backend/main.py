import os
import json
import asyncio
import base64
import traceback
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from google import genai
from google.genai import types

# Local Imports
import config
import security
import cv2
import numpy as np

# --- 1. AUTHENTICATION ---
key_path = os.path.abspath("serviceAccountKey.json")
if os.path.exists(key_path):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path
    print(f"🔑 Auth: Loaded credentials from {key_path}")
else:
    print("❌ ERROR: serviceAccountKey.json NOT FOUND.")

app = FastAPI()

# --- 2. INITIALIZE CLIENT ---
client = None
try:
    if config.USE_VERTEX_AI:
        print(f"🌍 Connecting to Vertex AI ({config.GOOGLE_CLOUD_LOCATION})...")
        client = genai.Client(
            vertexai=True,
            project=config.GOOGLE_CLOUD_PROJECT,
            location=config.GOOGLE_CLOUD_LOCATION,
            http_options=types.HttpOptions(api_version='v1beta1')
        )
        print("✅ Vertex AI Client Ready")
    else:
        client = genai.Client(
            api_key=config.GEMINI_API_KEY,
            http_options=types.HttpOptions(api_version='v1beta1')
        )
except Exception as e:
    print(f"❌ Client Init Error: {e}")

# Initialize Security
security.init_firebase()


def calculate_motion_score(current_bytes, previous_frame_gray):
    """Simple motion detection."""
    nparr = np.frombuffer(current_bytes, np.uint8)
    current_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if current_frame is None:
        return 0.0, previous_frame_gray

    gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (100, 100))
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if previous_frame_gray is None:
        return 100.0, gray

    frame_delta = cv2.absdiff(previous_frame_gray, gray)
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
    return (np.count_nonzero(thresh) / thresh.size) * 100, gray


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    if not await security.verify_token(websocket):
        await websocket.close(code=4001)
        return

    mode = websocket.query_params.get("mode", "default")
    print(f"✅ Client Connected ({mode})")

    if not client:
        await websocket.close(code=1011)
        return

    try:
        # --- 3. ZERO-CONFIG CONNECTION ---
        async with client.aio.live.connect(
            model=config.GEMINI_MODEL,
            config=None
        ) as session:
            print("🚀 Connected to Gemini Live session")

            # --- 4. MANUAL SETUP PROMPT ---
            system_instruction = config.PERSONAS.get(
                mode, config.PERSONAS["safety"])
            greeting = config.GREETINGS.get(mode, "System Online.")

            await session.send(
                input=f"SYSTEM_INSTRUCTION: {system_instruction}\nTASK: Say '{greeting}' to confirm you are ready.",
                end_of_turn=True
            )

            # --- Task A: AI -> Mobile ---
            async def receive_from_gemini():
                try:
                    while True:
                        async for response in session.receive():
                            if response.server_content and response.server_content.model_turn:
                                for part in response.server_content.model_turn.parts:
                                    if part.text:
                                        text = part.text.strip()
                                        if text:
                                            print(f"🤖 Aura: {text}")
                                            priority = "normal"
                                            if mode == "safety" and "[CRITICAL]" in text:
                                                priority = "high"
                                            await websocket.send_text(json.dumps({
                                                "cmd": "speak",
                                                "text": text,
                                                "priority": priority
                                            }))
                except Exception as e:
                    print(f"⚠️ Receive Error: {e}")

            # --- Task B: Mobile -> AI ---
            async def receive_from_mobile():
                import time
                last_processed_frame_gray = None
                last_trigger_time = 0
                COOLDOWN = 2.0

                while True:
                    try:
                        data = await websocket.receive_text()
                        payload = json.loads(data)

                        # --- IMAGE HANDLING (CORRECT) ---
                        if "image" in payload:
                            img_bytes = base64.b64decode(payload["image"])

                            # 1. Create Blob from raw bytes
                            image_blob = types.Blob(
                                data=img_bytes,
                                mime_type="image/jpeg"
                            )

                            # 2. Wrap in Part
                            image_part = types.Part(inline_data=image_blob)

                            # 3. CREATE CONTENT OBJECT WITH THE PART
                            # This is the crucial step - wrap Part in Content
                            image_content = types.Content(parts=[image_part])

                            # 4. Send the Content object (not Part directly)
                            await session.send(input=image_content, end_of_turn=False)

                            # Motion Logic
                            score, last_processed_frame_gray = calculate_motion_score(
                                img_bytes, last_processed_frame_gray
                            )

                            if score > 5.0 and (time.time() - last_trigger_time > COOLDOWN):
                                last_trigger_time = time.time()
                                print(f"🚀 Motion ({score:.0f}%) -> Analyzing")
                                await session.send(input="Describe hazards.", end_of_turn=True)

                        # User Text
                        if "text" in payload:
                            print(f"🗣️ User: {payload['text']}")
                            await session.send(input=payload['text'], end_of_turn=True)

                    except WebSocketDisconnect:
                        print("📱 Mobile Disconnected")
                        break
                    except Exception as e:
                        print(f"⚠️ Mobile Loop Error: {e}")
                        traceback.print_exc()
                        break

            await asyncio.gather(receive_from_gemini(), receive_from_mobile())

    except WebSocketDisconnect:
        print("❌ WebSocket Disconnected")
    except Exception as e:
        print(f"🔥 Session Error: {e}")
        traceback.print_exc()
    finally:
        try:
            await websocket.close()
        except:
            pass
