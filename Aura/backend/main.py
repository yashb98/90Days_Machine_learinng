import os
import json
import re
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


app = FastAPI()

# 1. Initialize Security (Firebase)
security.init_firebase()

# 2. Initialize Gemini Client (New SDK)
client = None
if config.GEMINI_API_KEY:
    # 'http_options' allows us to target v1alpha for the experimental 2.0 model
    client = genai.Client(
        api_key=config.GEMINI_API_KEY,
        http_options={"api_version": "v1alpha"}
    )
    print("✅ Gemini Client Initialized (New SDK)")
else:
    print("⚠️ WARNING: GEMINI_API_KEY is missing.")


def calculate_motion_score(current_bytes, previous_frame_gray):
    """
    Returns a score (0-100) representing how much the scene changed.
    Also returns the processed current frame for the next comparison.
    """
    # 1. Decode bytes to OpenCV Image
    nparr = np.frombuffer(current_bytes, np.uint8)
    current_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if current_frame is None:
        return 0.0, previous_frame_gray

    # 2. Preprocessing (Gray -> Resize -> Blur)
    # We resize to a small thumbnail (e.g., 64x64) for extreme speed
    gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (100, 100))
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if previous_frame_gray is None:
        # First frame ever, no comparison possible yet
        return 100.0, gray

    # 3. Calculate Difference (Absolute Diff)
    frame_delta = cv2.absdiff(previous_frame_gray, gray)
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]

    # 4. Calculate Motion Score (Percentage of changed pixels)
    # count non-zero pixels / total pixels
    change_score = (np.count_nonzero(thresh) / thresh.size) * 100

    return change_score, gray


@app.get("/")
async def health_check():
    return {"status": "online", "message": "Aura Brain is Listening"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    # 1. Verify Auth
    if not await security.verify_token(websocket):
        await websocket.close(code=4001)
        print("⛔ Connection Rejected: Invalid Token")
        return

    # 2. Determine Mode
    mode = websocket.query_params.get("mode", "safety")
    print(f"✅ Client Connected ({mode})")

    if not client:
        print("❌ Server Error: Gemini Client not ready")
        await websocket.close(code=1011)
        return

    # 3. Configure Session using New SDK Types
    selected_instruction = config.PERSONAS.get(mode, config.PERSONAS["safety"])

    # Configuration object for Live API
    live_config = types.LiveConnectConfig(
        # Change to ["AUDIO"] if you want raw audio back
        response_modalities=["TEXT"],
        system_instruction=types.Content(
            parts=[types.Part(text=selected_instruction)]
        ),
    )

    try:
        # 4. Connect to Gemini Live
        async with client.aio.live.connect(model=config.GEMINI_MODEL, config=live_config) as session:
            print("🚀 Connected to Gemini Live")

            # Send Greeting
            greeting_text = config.GREETINGS.get(mode, "System Online.")
            await session.send(input=greeting_text, end_of_turn=False)

            # --- Task A: Receive from Gemini (AI -> Mobile) ---
            async def receive_from_gemini():
                # Buffer to hold text until we get a full sentence
                sentence_buffer = ""

                try:
                    while True:
                        async for response in session.receive():
                            server_content = response.server_content
                            if server_content is None:
                                continue

                            model_turn = server_content.model_turn
                            if model_turn is None:
                                continue

                            for part in model_turn.parts:
                                if part.text:
                                    # 1. Add new chunk to the buffer
                                    sentence_buffer += part.text

                                    # 2. Check if we have a complete sentence ending (. ? ! or newline)
                                    # This Regex looks for punctuation followed by space or end of string
                                    if re.search(r'[.!?\n]', sentence_buffer):

                                        # Clean up the text (remove ** bolding markers which confuse TTS)
                                        clean_text = sentence_buffer.replace(
                                            "*", "").strip()

                                        if clean_text:
                                            print(
                                                f"✅ [Response] Full Sentence: '{clean_text}'")

                                            # Priority Logic
                                            priority = "normal"
                                            if mode == "safety" and "[CRITICAL]" in clean_text:
                                                priority = "high"

                                            # Send the FULL SENTENCE to mobile
                                            payload = {
                                                "cmd": "speak",
                                                "text": clean_text,
                                                "priority": priority
                                            }
                                            await websocket.send_text(json.dumps(payload))

                                        # 3. Clear buffer after sending
                                        sentence_buffer = ""

                except Exception as e:
                    print(f" Gemini Receive Error: {e}")
                    traceback.print_exc()

            # --- Task B: Receive from Mobile (Mobile -> AI) ---
            async def receive_from_mobile():
                import time  # Ensure time is imported
                # create a folder to save images (debugging)
                if not os.path.exists("debug_frames"):
                    os.makedirs("debug_frames")

                frames_saved_count = 0

                # Motion State
                last_processed_frame_gray = None
                last_trigger_time = 0

                # CONFIGURATION
                # How much must the scene change to trigger AI? (0-100)
                # 2.0 = Sensitive (detects hand waving)
                # 10.0 = Coarse (detects walking/turning)
                MOTION_THRESHOLD = 15.0

                # Minimum seconds between auto-descriptions (prevents spam)
                COOLDOWN_SECONDS = 5.0

                while True:
                    try:
                        data = await websocket.receive_text()
                        payload = json.loads(data)

                        # 1. VIDEO FRAMES
                        if "image" in payload:
                            frame_id = payload.get("frame_id", "Unknown")
                            image_bytes = base64.b64decode(payload["image"])

                            # --- A. ALWAYS SEND TO GEMINI BUFFER (Turn = False) ---
                            # We always feed the eyes, so when we ask, it knows history.
                            print(
                                f"🧐 [Backend] Streaming Frame #{frame_id}...")
                            # Save the first 5 frames to verify quality/rotation
                            if frames_saved_count < 5:
                                filename = f"debug_frames/frame_{frame_id}.jpg"
                                with open(filename, "wb") as f:
                                    f.write(image_bytes)
                                print(
                                    f"[Debug] Saved {filename} (Check this file!)")
                                frames_saved_count += 1

                            # send to Gemini Buffer
                            await session.send(
                                input=types.LiveClientRealtimeInput(
                                    media_chunks=[
                                        types.Blob(

                                            data=image_bytes,
                                            mime_type="image/jpeg"
                                        )
                                    ]
                                ),
                                end_of_turn=False
                            )

                            # --- B. CALCULATE MOTION (The Trigger) ---
                            motion_score, last_processed_frame_gray = calculate_motion_score(
                                image_bytes,
                                last_processed_frame_gray
                            )

                            current_time = time.time()
                            time_since_last = current_time - last_trigger_time

                            # Log the motion score for debugging
                            # print(f"📊 Motion Score: {motion_score:.2f}%")

                            # --- C. DECIDE TO TRIGGER AI ---
                            if motion_score > MOTION_THRESHOLD and time_since_last > COOLDOWN_SECONDS:
                                print(
                                    f"🚀 [Backend] MOTION DETECTED ({motion_score:.1f}%) - Triggering AI")

                                last_trigger_time = current_time

                                # Custom Prompt based on Mode
                                auto_prompt = "Describe what is in view."
                                if mode == "safety":
                                    auto_prompt = (" Describe only clear obstacles or moving objects in this frame. "
                                                   "If unsure, say 'No clear obstacle visible.'"
                                                   )
                                elif mode == "reading":
                                    auto_prompt = (
                                        "Read only large, legible printed signs or documents. "
                                        "Ignore any text shown on screens or UI elements."
                                    )

                                # FORCE RESPONSE (End Turn = True)
                                await session.send(input=auto_prompt, end_of_turn=True)

                        # 2. VOICE COMMANDS (Manual Override)
                        if "text" in payload:
                            user_command = payload['text']
                            print(
                                f"🗣️ [Command] User Interrupt: {user_command}")

                            # Reset cooldown so we can trigger again immediately after this
                            last_trigger_time = 0

                            await session.send(input=user_command, end_of_turn=True)

                    except WebSocketDisconnect:
                        print("📱 Mobile Disconnected")
                        break
                    except Exception as e:
                        print(f"⚠️ Error: {type(e).__name__}: {e}")
                        traceback.print_exc()
                        continue
            # Run both tasks concurrently
            await asyncio.gather(receive_from_gemini(), receive_from_mobile())

    except WebSocketDisconnect:
        print("❌ WebSocket Disconnected")
    except Exception as e:
        print(f"🔥 Critical Logic Error: {e}")
        traceback.print_exc()
    finally:
        try:
            await websocket.close()
        except:
            pass
