import cv2
import numpy as np
import google.generativeai as genai
import os
import base64
import json
import asyncio
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from dotenv import load_dotenv
import PIL.Image
import io

# --- 1. SETUP ---
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    print("ERROR: GEMINI_API_KEY not found. Server will start but AI will fail.")
else:
    genai.configure(api_key=API_KEY)
    print("Gemini Configured")

app = FastAPI()

# --- 2. VISION LOGIC (The Eyes) ---


def has_scene_changed(prev_frame, curr_frame, threshold=40):
    """
    Returns True if the scene has shifted significantly (The 'Barge-In' trigger).
    """
    if prev_frame is None:
        return False

    # Convert to grayscale for speed
    prev_g = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_g = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    # Calculate difference
    diff = cv2.absdiff(prev_g, curr_g)
    score = np.mean(diff)

    return score > threshold

# --- 3. AI LOGIC (The Brain with Resiliency) ---


async def generate_response_safe(image_bytes, max_retries=3):
    """
    Calls Gemini with Exponential Backoff for poor connections.
    """
    base_delay = 1

    for attempt in range(max_retries):
        try:
            # 1. Prepare Image
            image = PIL.Image.open(io.BytesIO(image_bytes))

            # 2. Call AI (Using synchronous call in async wrapper if needed,
            # or use the async client if available. For safety, we wrap this.)
            model = genai.GenerativeModel('gemini-2.0-flash-exp')
            response = await asyncio.to_thread(
                model.generate_content,
                ["Describe this scene strictly for a blind person. Be brief.", image]
            )
            return response.text

        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(base_delay * (2 ** attempt))
            else:
                return "I am having trouble connecting to the brain."

# --- 4. WEBSOCKET SERVER (The Nervous System) ---


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Mobile Client Connected")

    prev_frame = None
    is_processing = False

    # AUTO-SCAN TIMER SETUP
    last_scan_time = time.time()
    SCAN_INTERVAL = 5.0  # Seconds between auto-voice descriptions

    try:
        while True:
            # A. Receive Data (Non-blocking usually, but here we wait for frame)
            # In a real async loop, we'd use asyncio.wait_for, but for simplicity:
            data = await websocket.receive_text()
            payload = json.loads(data)

            current_time = time.time()

            # CASE 1: Incoming Video Frame
            if "image" in payload:
                # Decode Frame
                image_data = base64.b64decode(payload["image"])
                np_arr = np.frombuffer(image_data, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                if frame is None:
                    continue
                cv2.imwrite("debug_recieved_frame.jpg", frame)
                # 1. CHECK FOR BARGE-IN (Interruption)
                if has_scene_changed(prev_frame, frame):
                    if is_processing:
                        await websocket.send_json({"cmd": "interrupt"})
                        print("⚡ INTERRUPT SENT (Scene Changed)")
                        is_processing = False

                prev_frame = frame

                # 2. AUTO-TRIGGER (The Fix!)
                # If 5 seconds passed AND we aren't already talking
                if (current_time - last_scan_time > SCAN_INTERVAL) and not is_processing:
                    print("Auto-Scanning...")
                    is_processing = True
                    last_scan_time = current_time  # Reset timer

                    # Notify App: "Thinking"
                    await websocket.send_json({"cmd": "status", "state": "thinking"})

                    # Encode and Send to AI
                    success, encoded_img = cv2.imencode('.jpg', frame)
                    if success:
                        # Run AI in thread to not block the websocket loop
                        response_text = await generate_response_safe(encoded_img.tobytes())

                        # Send Audio Command
                        await websocket.send_json({
                            "cmd": "speak",
                            "text": response_text
                        })
                        print(f"Sent: {response_text}")

                    is_processing = False
                    await websocket.send_json({"cmd": "status", "state": "ready"})

            # CASE 2: Manual Button Press (Still works if you have a button)
            if "event" in payload and payload["event"] == "scan":
                # (Logic handled by auto-scan now, but you can keep specific triggers here)
                pass

    except WebSocketDisconnect:
        print("Client Disconnected")
    except Exception as e:
        print(f"Server Error: {e}")
