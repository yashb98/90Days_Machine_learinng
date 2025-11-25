import json
import asyncio
import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
from google import genai
from dotenv import load_dotenv

# Load API Key
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

app = FastAPI()

# --- SECURITY: Firebase Auth Placeholder ---
# import firebase_admin
# from firebase_admin import credentials, auth
# cred = credentials.Certificate("serviceAccountKey.json")
# firebase_admin.initialize_app(cred)


async def verify_token(websocket: WebSocket):
    # token = websocket.query_params.get("token")
    # if not token: return False
    # try:
    #     decoded_token = auth.verify_id_token(token)
    #     return True
    # except:
    #     return False
    return True  # Bypassing security for initial pipeline test

# --- GEMINI CLIENT SETUP ---
client = genai.Client(api_key=API_KEY, http_options={"api_version": "v1alpha"})
MODEL = "models/gemini-2.0-flash-exp"
CONFIG = {"response_modalities": ["AUDIO"]}


@app.get("/")
async def health_check():
    return {"status": "online", "message": "Aura Brain is Listening"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    # 1. Security Check
    if not await verify_token(websocket):
        await websocket.close(code=4001)
        print("Security Alert: Invalid Token")
        return

    print("Client Connected (Mobile)")

    try:
        # 2. Open Connection to Gemini (The Proxy)
        async with client.aio.live.connect(model=MODEL, config=CONFIG) as session:
            print("Connected to Gemini Live")

            # Task A: Receive from Gemini -> Send to Mobile (Audio)
            async def receive_from_gemini():
                while True:
                    async for response in session.receive():
                        if response.data:
                            # Forward raw audio chunk to mobile
                            # Gemini sends PCM data, we wrap it in JSON
                            payload = {
                                "audio": os.base64.b64encode(response.data).decode('utf-8')
                            }
                            await websocket.send_text(json.dumps(payload))

            # Task B: Receive from Mobile -> Send to Gemini (Video)
            async def receive_from_mobile():
                while True:
                    data = await websocket.receive_text()
                    if not data:
                        continue

                    payload = json.loads(data)

                    # If we have an image, send it to Gemini
                    if "image" in payload:
                        # Decode Base64 to bytes
                        image_bytes = os.base64.b64decode(payload["image"])
                        # Send to Gemini
                        await session.send(input={"mime_type": "image/jpeg", "data": image_bytes}, end_of_turn=True)

            # Run both loops simultaneously
            await asyncio.gather(receive_from_gemini(), receive_from_mobile())

    except WebSocketDisconnect:
        print("Mobile Disconnected")
    except Exception as e:
        print(f" Error: {e}")
    finally:
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.close()
