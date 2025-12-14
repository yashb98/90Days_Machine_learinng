import os
import json
import asyncio
import base64
import traceback
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from google import genai
from google.genai import types
from google.cloud import vision # <--- NEW: Vision API

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

# --- 2. INITIALIZE CLIENTS ---
# A. Gemini Client (For Understanding & Speech)
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
    else:
        client = genai.Client(api_key=config.GEMINI_API_KEY, http_options=types.HttpOptions(api_version='v1beta1'))
    print("✅ Gemini Live Client Ready")
except Exception as e:
    print(f"❌ Gemini Init Error: {e}")

# B. Vision Client (For 100% OCR Accuracy)
try:
    vision_client = vision.ImageAnnotatorClient()
    print("✅ Cloud Vision API Ready")
except Exception as e:
    print(f"❌ Vision API Init Error: {e}")

# Initialize Security
security.init_firebase()

# --- HELPER FUNCTIONS ---
def calculate_motion_score(current_bytes, previous_frame_gray):
    """Detects if the camera is steady enough to read."""
    nparr = np.frombuffer(current_bytes, np.uint8)
    current_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if current_frame is None: return 100.0, previous_frame_gray # Assume motion if error

    gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (100, 100))
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if previous_frame_gray is None: return 100.0, gray

    frame_delta = cv2.absdiff(previous_frame_gray, gray)
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
    score = (np.count_nonzero(thresh) / thresh.size) * 100
    return score, gray

async def scan_document_with_vision(image_bytes):
    """Sends image to Google Cloud Vision API for high-precision OCR."""
    try:
        image = vision.Image(content=image_bytes)
        # We run this in a thread because Vision API is synchronous
        response = await asyncio.get_event_loop().run_in_executor(
            None, vision_client.document_text_detection, image
        )
        if response.error.message:
            print(f"⚠️ OCR Error: {response.error.message}")
            return None
        return response.full_text_annotation.text
    except Exception as e:
        print(f"⚠️ OCR Exception: {e}")
        return None

# --- MAIN WEBSOCKET ---
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    if not await security.verify_token(websocket):
        await websocket.close(code=4001)
        return

    mode = websocket.query_params.get("mode", "default")
    print(f"✅ Client Connected (Mode: {mode})")

    if not client:
        await websocket.close(code=1011)
        return

    try:
        # Configure Session
        # We ALWAYS need Audio output, regardless of mode
        live_config = types.LiveConnectConfig(response_modalities=["AUDIO"])

        async with client.aio.live.connect(model=config.GEMINI_MODEL, config=live_config) as session:
            print(f"🚀 Gemini Session Started ({config.GEMINI_MODEL})")

            # Custom System Instructions based on Mode
            if mode == "reading":
                instruction = "You are a reading assistant. I will send you text that I scanned. Read it out loud clearly and naturally. If it is a check, announce the amount and sender clearly."
                greeting = "Reading mode ready. Hold the document steady."
            else:
                instruction = config.PERSONAS.get(mode, config.PERSONAS["safety"])
                greeting = config.GREETINGS.get(mode, "System Online.")

            await session.send(input=f"System Instruction: {instruction}\n\nTask: Say '{greeting}'", end_of_turn=True)

            # --- Task A: Gemini -> Mobile (Audio/Text) ---
            async def receive_from_gemini():
                try:
                    while True:
                        async for response in session.receive():
                            if response.server_content and response.server_content.model_turn:
                                for part in response.server_content.model_turn.parts:
                                    if part.text:
                                        print(f"🤖 Aura: {part.text.strip()}")
                                        await websocket.send_text(json.dumps({
                                            "cmd": "speak",
                                            "text": part.text.strip()
                                        }))
                                    elif part.inline_data:
                                        pass 
                except Exception as e:
                    print(f"⚠️ Gemini Receive Error: {e}")

            # --- Task B: Mobile -> Backend (Vision Logic) ---
            async def receive_from_mobile():
                import time
                last_processed_frame_gray = None
                last_trigger_time = 0
                
                # Reading mode needs STABILITY (low motion), Safety mode needs CHANGE (high motion)
                COOLDOWN = 3.0 if mode == "reading" else 2.0 

                while True:
                    try:
                        data = await websocket.receive_text()
                        payload = json.loads(data)

                        if "image" in payload:
                            b64_image = payload["image"]
                            img_bytes = base64.b64decode(b64_image)

                            # 1. Calculate Motion
                            motion_score, last_processed_frame_gray = calculate_motion_score(
                                img_bytes, last_processed_frame_gray
                            )

                            current_time = time.time()
                            
                            # --- LOGIC BRANCHING BASED ON MODE ---
                            
                            # A. READING MODE (High Precision OCR)
                            if mode == "reading":
                                # Trigger ONLY if camera is steady (low motion) and cooldown passed
                                if motion_score < 1.0 and (current_time - last_trigger_time > COOLDOWN):
                                    print("📖 Camera Steady - Scanning Document...")
                                    last_trigger_time = current_time
                                    
                                    # 1. Get raw text from Vision API (The "Perfect" Reader)
                                    ocr_text = await scan_document_with_vision(img_bytes)
                                    
                                    if ocr_text and len(ocr_text) > 5:
                                        print(f"📄 OCR Success: {ocr_text[:50]}...")
                                        # 2. Feed text to Gemini to speak naturally
                                        prompt = f"I just scanned this text: '{ocr_text}'. Read it to me clearly."
                                        await session.send(input=prompt, end_of_turn=True)
                                    else:
                                        print("...No text found.")

                            # B. SAFETY/NAVIGATION MODE (General Vision)
                            else:
                                # Stream vision to Gemini directly
                                await session.send(
                                    input={"inline_data": {"mime_type": "image/jpeg", "data": b64_image}},
                                    end_of_turn=False
                                )
                                # Trigger if motion is HIGH (Something is happening)
                                if motion_score > 5.0 and (current_time - last_trigger_time > COOLDOWN):
                                    print(f"🚀 Motion ({motion_score:.0f}%) -> Analyzing Hazard...")
                                    last_trigger_time = current_time
                                    await session.send(input="Describe hazards.", end_of_turn=True)

                        if "text" in payload:
                            print(f"🗣️ User: {payload['text']}")
                            await session.send(input=payload['text'], end_of_turn=True)

                    except WebSocketDisconnect:
                        break
                    except Exception as e:
                        print(f"⚠️ Loop Error: {e}")
                        traceback.print_exc()
                        break

            await asyncio.gather(receive_from_gemini(), receive_from_mobile())

    except Exception as e:
        print(f"🔥 Session Error: {e}")
    finally:
        try:
            await websocket.close()
        except:
            pass