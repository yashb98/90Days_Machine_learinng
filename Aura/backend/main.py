import os
import json
import re
import asyncio
import base64
import traceback
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from google import genai
from google.genai.types import HttpOptions, Part, LiveConnectConfig, Content

# Local Imports
import config
import security
import cv2
import numpy as np

app = FastAPI()

# --- 1. SETUP AUTHENTICATION ---
# Point to your service account key so Vertex AI can authenticate
# This MUST be set before initializing the client if using Vertex AI
if os.path.exists("serviceAccountKey.json"):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "serviceAccountKey.json"
    print("✅ Auth: Using serviceAccountKey.json for Vertex AI")

# Initialize Firebase (Legacy/Mobile Auth)
security.init_firebase()

# --- 2. INITIALIZE GEMINI CLIENT (Unified SDK) ---
client = None
try:
    if config.USE_VERTEX_AI:
        # VERTEX AI INITIALIZATION
        print(
            f"🌍 Connecting to Vertex AI (Project: {config.GOOGLE_CLOUD_PROJECT})...")
        client = genai.Client(
            vertexai=True,
            project=config.GOOGLE_CLOUD_PROJECT,
            location=config.GOOGLE_CLOUD_LOCATION,
            http_options=HttpOptions(api_version="v1alpha")
        )
        print("✅ Vertex AI Client Ready")

    elif config.GEMINI_API_KEY:
        # AI STUDIO INITIALIZATION (Fallback)
        print("🔑 Connecting to Google AI Studio (API Key)...")
        client = genai.Client(
            api_key=config.GEMINI_API_KEY,
            http_options=HttpOptions(api_version="v1alpha")
        )
        print("✅ AI Studio Client Ready")

    else:
        print("⚠️ WARNING: No Credentials found (Set USE_VERTEX_AI or GEMINI_API_KEY)")

except Exception as e:
    print(f"❌ Client Init Error: {e}")


def calculate_motion_score(current_bytes, previous_frame_gray):
    """
    Returns a score (0-100) representing how much the scene changed.
    """
    nparr = np.frombuffer(current_bytes, np.uint8)
    current_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if current_frame is None:
        return 0.0, previous_frame_gray

    # Preprocessing
    gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (100, 100))
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if previous_frame_gray is None:
        return 100.0, gray

    # Difference Calculation
    frame_delta = cv2.absdiff(previous_frame_gray, gray)
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
    change_score = (np.count_nonzero(thresh) / thresh.size) * 100

    return change_score, gray


@app.post("/vision/default")
async def vision_default(image: UploadFile = File(...)):
    """Simple REST endpoint for single-image analysis"""
    data = await image.read()
    image_part = Part.from_data(
        data=data, mime_type=image.content_type or "image/jpeg")

    resp = client.models.generate_content(
        model=config.GEMINI_MODEL,
        contents=[
            "Describe exactly what is in this image in one short sentence.", image_part],
    )
    return {"description": resp.text}


@app.get("/")
async def health_check():
    return {
        "status": "online",
        "backend": "Vertex AI" if config.USE_VERTEX_AI else "AI Studio",
        "model": config.GEMINI_MODEL
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    # 1. Verify Auth
    if not await security.verify_token(websocket):
        await websocket.close(code=4001)
        print("⛔ Connection Rejected: Invalid Token")
        return

    mode = websocket.query_params.get("mode", "default")
    print(f"✅ Client Connected ({mode})")

    if not client:
        await websocket.close(code=1011)
        return

    # 2. Configure Session (Vertex AI & AI Studio use the same config structure)
    if mode == "default":
        live_config = LiveConnectConfig(response_modalities=["TEXT"])
    else:
        selected_instruction = config.PERSONAS.get(
            mode, config.PERSONAS["safety"])
        live_config = LiveConnectConfig(
            response_modalities=["TEXT"],
            system_instruction=Content(
                parts=[Part(text=selected_instruction)]),
        )

    try:
        # 3. Connect to Gemini Live
        async with client.aio.live.connect(model=config.GEMINI_MODEL, config=live_config) as session:
            print("🚀 Connected to Gemini Live session")

            greeting_text = config.GREETINGS.get(mode, "Aura online.")
            await session.send(input=greeting_text, end_of_turn=False)

            # --- Task A: AI -> Mobile ---
            async def receive_from_gemini():
                sentence_buffer = ""
                try:
                    while True:
                        async for response in session.receive():
                            server_content = response.server_content
                            if server_content is None or server_content.model_turn is None:
                                continue

                            for part in server_content.model_turn.parts:
                                if part.text:
                                    sentence_buffer += part.text

                                    # Detect full sentences for smoother TTS
                                    if re.search(r'[.!?\n]', sentence_buffer):
                                        clean_text = sentence_buffer.replace(
                                            "*", "").strip()
                                        if clean_text:
                                            print(f"🤖 Aura: {clean_text}")

                                            # Logic to filter fake [CRITICAL] alerts
                                            priority = "normal"
                                            if mode == "safety" and "[critical]" in clean_text.lower():
                                                priority = "high"

                                            await websocket.send_text(json.dumps({
                                                "cmd": "speak",
                                                "text": clean_text,
                                                "priority": priority
                                            }))
                                        sentence_buffer = ""
                except Exception as e:
                    print(f"⚠️ Gemini Receive Error: {e}")

            # --- Task B: Mobile -> AI ---
            async def receive_from_mobile():
                import time
                frames_processed = 0
                last_processed_frame_gray = None
                last_trigger_time = 0

                MOTION_THRESHOLD = 5.0
                COOLDOWN_SECONDS = 3.5  # Slightly faster for Vertex

                while True:
                    try:
                        data = await websocket.receive_text()
                        payload = json.loads(data)

                        if "image" in payload:
                            image_bytes = base64.b64decode(payload["image"])

                            # 1. Stream Image to Context (Turn=False)
                            # Vertex handles high-throughput video tokens well
                            await session.send(
                                input=Part.from_data(
                                    data=image_bytes, mime_type="image/jpeg"),
                                end_of_turn=False
                            )

                            # 2. Motion Trigger Logic
                            motion_score, last_processed_frame_gray = calculate_motion_score(
                                image_bytes, last_processed_frame_gray
                            )

                            if motion_score > MOTION_THRESHOLD and (time.time() - last_trigger_time > COOLDOWN_SECONDS):
                                print(
                                    f"🚀 Motion ({motion_score:.1f}%) -> Triggering AI")
                                last_trigger_time = time.time()

                                prompt_text = "Describe what you see."
                                if mode == "safety":
                                    prompt_text = "Identify obstacles."

                                await session.send(input=prompt_text, end_of_turn=True)

                        if "text" in payload:
                            print(f"🗣️ User: {payload['text']}")
                            last_trigger_time = 0  # Reset cooldown
                            await session.send(input=payload['text'], end_of_turn=True)

                    except WebSocketDisconnect:
                        print("📱 Mobile Disconnected")
                        break
                    except Exception as e:
                        print(f"⚠️ Mobile Receive Error: {e}")
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
