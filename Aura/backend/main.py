import os
import asyncio
import base64
import traceback
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

import config
import security
from services.ai_service import AIService
from services.vision_service import VisionService  # NEW
from services.socket_manager import WebSocketManager
from utils.image_ops import calculate_motion_score
from services.location_service import LocationService, GeoPose
from services.memory_service import MemoryService


# Tell google cloud where the key is
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "serviceAccountKey.json"

# --- INIT ---
app = FastAPI()
security.init_firebase()

# Services
ai_service = AIService()
vision_service = VisionService()
socket_manager = WebSocketManager()
loc_service = LocationService()
memory_service = MemoryService(loc_service)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    # 1. Auth Check
    is_valid = await security.verify_token(websocket)
    if not is_valid:
        await websocket.close(code=4001)
        return

    try:
        mode = await socket_manager.connect(websocket)

        # 2. Connect to Vertex AI Live Session
        async with ai_service.connect() as session:
            print(f"🚀 Gemini Live Session Started (Mode: {mode})")
            await ai_service.send_setup_prompt(session, mode)

            # --- A. Gemini -> Mobile Loop ---
            async def receive_from_gemini():
                try:
                    while True:
                        async for response in session.receive():
                            # Handle Text Response
                            if response.server_content and response.server_content.model_turn:
                                for part in response.server_content.model_turn.parts:
                                    if part.text:
                                        print(f"🤖 AI: {part.text}")
                                        await socket_manager.send_response(websocket, part.text)

                            # Handle Audio Response (if modality is AUDIO)
                            if response.server_content and response.server_content.model_turn:
                                for part in response.server_content.model_turn.parts:
                                    if part.inline_data:
                                        # Forward raw audio bytes to mobile
                                        await websocket.send_bytes(part.inline_data.data)

                except Exception as e:
                    print(f"⚠️ Receive Error: {e}")

            # --- B. Mobile -> Gemini Loop ---
            async def receive_from_mobile():
                last_processed_frame_gray = None
                last_known_pose = None  # Track where the user is

                while True:
                    try:
                        data = await websocket.receive_text()
                        payload = await socket_manager.parse_message(data)

                        # 1. Handle ARCore Geospatial Pose
                        if "pose" in payload:
                            p = payload["pose"]
                            last_known_pose = GeoPose(
                                lat=p.get("lat", 0.0),
                                lng=p.get("lng", 0.0),
                                heading=p.get("heading", 0.0)
                            )

                        # 2. Handle Image Processing
                        if "image" in payload:
                            img_bytes = base64.b64decode(payload["image"])

                            context_str = ""
                            if last_known_pose:
                                # what is important here?
                                context_str = await memory_service.recall(
                                    "hazards or important features",
                                    last_known_pose
                                )

                            # Inject memory context into sessiopn if found
                            if context_str:
                                print(f"Memory Recall: {context_str}")
                                await ai_service.send_text(session, f"CONTEXT: {context_str}")

                            # SPECIAL CASE: Cloud Vision for Text
                            # If user explicitly asks to read, or mode is "reader"
                            if payload.get("intent") == "read_text":
                                print("📖 Route: Cloud Vision API (OCR)")
                                text = vision_service.detect_text(img_bytes)
                                await socket_manager.send_response(websocket, f"Reading: {text}")
                                continue  # Skip sending to Gemini to save latency/cost

                            # STANDARD CASE: Send to Vertex AI
                            await ai_service.send_image_frame(session, img_bytes)

                            # Motion Logic (Local Optimization)
                            score, last_processed_frame_gray = calculate_motion_score(
                                img_bytes, last_processed_frame_gray
                            )
                            if score > 15.0:
                                await ai_service.send_text(session, "Significant motion detected. Identify hazards.")

                        # 3. Handle Voice/Text Input
                        if "text" in payload:
                            print(f"🗣️ User: {payload['text']}")
                            text = payload['text']

                            if 'remember' in text.lower() and last_known_pose:
                                await memory_service.store_memory(text, last_known_pose)
                                await socket_manager.send_response(websocket, "Memory Saved")

                            await ai_service.send_text(session, payload['text'])

                    except WebSocketDisconnect:
                        print("📱 Mobile Disconnected")
                        socket_manager.disconnect(websocket)
                        break
                    except Exception as e:
                        print(f"⚠️ Mobile Loop Error: {e}")
                        break

            # Run loops
            await asyncio.gather(receive_from_gemini(), receive_from_mobile())

    except Exception as e:
        print(f"🔥 Critical Error: {e}")
        traceback.print_exc()
    finally:
        try:
            await websocket.close()
        except:
            pass
