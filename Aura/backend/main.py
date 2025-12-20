import os
import asyncio
import base64
import traceback
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

# Imports from our new Modules
import config
import security
from services.ai_service import AIService
from services.socket_manager import WebSocketManager
from utils.image_ops import calculate_motion_score

# --- AUTH SETUP ---
key_path = os.path.abspath("serviceAccountKey.json")
if os.path.exists(key_path):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path
else:
    print("❌ ERROR: serviceAccountKey.json NOT FOUND.")

# --- APP & SERVICES ---
app = FastAPI()
security.init_firebase()

# Initialize Singletons
ai_service = AIService()
socket_manager = WebSocketManager()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    try:
        # 1. Handle Mobile Connection
        mode = await socket_manager.connect(websocket)
    except ConnectionRefusedError:
        return  # Auth failed, socket closed

    try:
        # 2. Connect to AI
        async with ai_service.connect() as session:
            print("🚀 Connected to Gemini Live session")

            # 3. Initialize AI Persona
            await ai_service.send_setup_prompt(session, mode)

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
                                            # Determine priority logic here or in service
                                            priority = "high" if "[CRITICAL]" in text else "normal"
                                            await socket_manager.send_response(websocket, text, priority)
                except Exception as e:
                    print(f"⚠️ Gemini Rx Error: {e}")

            # --- Task B: Mobile -> AI ---
            async def receive_from_mobile():
                import time
                last_processed_frame_gray = None
                last_trigger_time = 0
                COOLDOWN = 2.0

                while True:
                    try:
                        data = await websocket.receive_text()
                        payload = await socket_manager.parse_message(data)

                        # Handle Image
                        if "image" in payload:
                            img_bytes = base64.b64decode(payload["image"])

                            # Send to AI using Service
                            await ai_service.send_image_frame(session, img_bytes)

                            # Calculate Motion using Utils
                            score, last_processed_frame_gray = calculate_motion_score(
                                img_bytes, last_processed_frame_gray
                            )

                            if score > 5.0 and (time.time() - last_trigger_time > COOLDOWN):
                                last_trigger_time = time.time()
                                print(f"🚀 Motion ({score:.0f}%) -> Triggering")
                                await ai_service.send_text(session, "Describe hazards.")

                        # Handle Text
                        if "text" in payload:
                            print(f"🗣️ User: {payload['text']}")
                            await ai_service.send_text(session, payload['text'])

                    except WebSocketDisconnect:
                        print("📱 Mobile Disconnected")
                        break
                    except Exception as e:
                        print(f"⚠️ Mobile Loop Error: {e}")
                        traceback.print_exc()
                        break

            # Run parallel loops
            await asyncio.gather(receive_from_gemini(), receive_from_mobile())

    except Exception as e:
        print(f"🔥 Session Error: {e}")
        traceback.print_exc()
    finally:
        try:
            await websocket.close()
        except:
            pass
