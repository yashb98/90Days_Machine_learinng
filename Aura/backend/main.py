import json
import asyncio
import os
import base64
import traceback
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from google import genai
import firebase_admin
from firebase_admin import credentials, auth

# Import settings
import config

app = FastAPI()

# --- 1. SAFE FIREBASE INIT ---
try:
    if not firebase_admin._apps:
        possible_paths = ["/app/serviceAccountKey.json",
                          "serviceAccountKey.json", "/secrets/serviceAccountKey.json"]
        key_path = next((p for p in possible_paths if os.path.exists(p)), None)

        if key_path:
            cred = credentials.Certificate(key_path)
            firebase_admin.initialize_app(cred)
            print(f"✅ Firebase Admin Initialized")
        else:
            print("⚠️ WARNING: serviceAccountKey.json not found.")
except Exception as e:
    print(f"❌ Firebase Init Error: {e}")

# --- 2. GEMINI CLIENT ---
client = None
try:
    if config.GEMINI_API_KEY:
        client = genai.Client(api_key=config.GEMINI_API_KEY, http_options={
                              "api_version": "v1alpha"})
        print("✅ Gemini Client Initialized")
    else:
        print("⚠️ WARNING: GEMINI_API_KEY is missing.")
except Exception as e:
    print(f"❌ Gemini Client Init Error: {e}")


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


@app.get("/")
async def health_check():
    return {"status": "online", "message": "Aura Brain is Listening"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    if not await verify_token(websocket):
        await websocket.close(code=4001)
        print("⛔ Connection Rejected")
        return

    # Determine Mode
    mode = websocket.query_params.get("mode", "safety")
    print(f"✅ Client Connected ({mode})")

    if not client:
        await websocket.close(code=1011)
        return

    # Configure Persona
    selected_instruction = config.PERSONAS.get(mode, config.PERSONAS["safety"])
    session_config = {
        "response_modalities": ["TEXT"],
        "system_instruction": selected_instruction,
    }

    try:
        async with client.aio.live.connect(model=config.GEMINI_MODEL, config=session_config) as session:
            print("🚀 Connected to Gemini Live")

            # Send Greeting
            greeting = config.GREETINGS.get(mode, "System Online.")
            await session.send(input=greeting, end_of_turn=True)

            # --- Task A: Gemini -> Mobile ---
            async def receive_from_gemini():
                try:
                    async for response in session.receive():
                        if response.text:
                            text_response = response.text
                            print(f"🤖 AI: {text_response}")

                            priority = "normal"
                            if mode == "safety" and "[CRITICAL]" in text_response:
                                priority = "high"

                            payload = {
                                "cmd": "speak", "text": text_response, "priority": priority}
                            await websocket.send_text(json.dumps(payload))
                except Exception as e:
                    print(f"❌ Gemini Error: {e}")

            # --- Task B: Mobile -> Gemini (ROBUST LOOP) ---
            async def receive_from_mobile():
                while True:  # Loop keeps running even if errors occur
                    try:
                        data = await websocket.receive_text()
                        payload = json.loads(data)

                        # 1. Voice Commands
                        if "text" in payload:
                            print(f"🗣️ Command: {payload['text']}")
                            await session.send(input=payload["text"], end_of_turn=True)

                        # 2. Images
                        if "image" in payload:
                            # print(".", end="", flush=True) # Heartbeat
                            image_bytes = base64.b64decode(payload["image"])
                            await session.send(input={"mime_type": "image/jpeg", "data": image_bytes}, end_of_turn=True)

                    except WebSocketDisconnect:
                        print("📱 Mobile Disconnected")
                        break  # Stop only on disconnect
                    except Exception as e:
                        print(f"⚠️ Frame Error (Ignored): {e}")
                        continue  # Skip bad frame, keep running!

            await asyncio.gather(receive_from_gemini(), receive_from_mobile())

    except WebSocketDisconnect:
        print("❌ Disconnected")
    except Exception as e:
        print(f"🔥 Critical Error: {e}")
        traceback.print_exc()
    finally:
        try:
            await websocket.close()
        except:
            pass
