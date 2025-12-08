import json
import asyncio
import base64
import traceback
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from google.genai import types  # <--- FIXED: Added this import

# Import from our new files
from config import GEMINI_MODEL, GEMINI_CONFIG
from security import init_firebase, verify_token
from gemini_client import get_gemini_client

app = FastAPI()

# Initialize Systems
init_firebase()
client = get_gemini_client()


@app.get("/")
async def health_check():
    return {"status": "online", "message": "Aura Brain is Listening"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    # 1. Security Check
    if not await verify_token(websocket):
        await websocket.close(code=4001)
        print("⛔ Connection Rejected: Invalid Token")
        return

    print("✅ Client Connected")

    if not client:
        await websocket.close(code=1011)
        return

    try:
        # FIXED: Use GEMINI_MODEL and GEMINI_CONFIG
        async with client.aio.live.connect(model=GEMINI_MODEL, config=GEMINI_CONFIG) as session:
            print("🚀 Connected to Gemini Live")

            # Wake Word
            await session.send(input="System Online. I will now describe your surroundings continuously.", end_of_turn=True)

            async def receive_from_gemini():
                try:
                    while True:
                        async for response in session.receive():
                            if response.text:
                                text_response = response.text.strip()
                                if text_response:
                                    print(f"🤖 Gemini: {text_response}")
                                    payload = {
                                        "cmd": "speak",
                                        "text": text_response,
                                        "priority": "high" if "[CRITICAL]" in text_response else "normal"
                                    }
                                    await websocket.send_text(json.dumps(payload))
                except Exception:
                    pass

            async def receive_from_mobile():
                try:
                    while True:
                        data = await websocket.receive_text()
                        payload = json.loads(data)

                        # 1. Voice Overrides (User asks specific question)
                        if "text" in payload:
                            print(f"🗣️ Command: {payload['text']}")
                            # Send text to Gemini (It will reply to this specific question)
                            await session.send(input=payload["text"], end_of_turn=True)

                        # 2. Continuous Vision Stream
                        if "image" in payload:
                            image_bytes = base64.b64decode(payload["image"])

                            # FIX: Send in two steps instead of a list

                            # Step 1: Send the context/prompt (Keep turn OPEN)
                            await session.send(input="Describe what you see.", end_of_turn=False)

                            # Step 2: Send the image data (Close turn -> Trigger Response)
                            await session.send(input={"mime_type": "image/jpeg", "data": image_bytes}, end_of_turn=True)

                except WebSocketDisconnect:
                    print("📱 Mobile Disconnected")
                    raise
                except Exception as e:
                    print(f"❌ Error: {e}")
                    traceback.print_exc()

            await asyncio.gather(receive_from_gemini(), receive_from_mobile())

    except WebSocketDisconnect:
        print("❌ Disconnected")
    except Exception as e:
        print(f"🔥 Error: {e}")
    finally:
        try:
            await websocket.close()
        except:
            pass
