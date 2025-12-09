import json
import asyncio
import base64
import traceback
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from google.genai import types

# Import from our new files
from config import MODEL, PERSONAS
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
        return

    # 2. Determine Mode from URL (ws://...?mode=reading)
    # Default to 'safety' if not specified
    mode = websocket.query_params.get("mode", "safety")
    selected_prompt = PERSONAS.get(mode, PERSONAS["safety"])

    print(f"✅ Client Connected in [{mode.upper()}] Mode")

    if not client:
        await websocket.close(code=1011)
        return

    # 3. Configure Session with Selected Persona
    session_config = {
        "response_modalities": ["TEXT"],
        "system_instruction": selected_prompt,
    }

    try:
        async with client.aio.live.connect(model=MODEL, config=session_config) as session:
            # Context-aware greeting
            greeting = {
                "safety": "Safety Watch Active.",
                "reading": "Text Mode. Show me text.",
                "scenery": "Scenery Mode. Ready to describe."
            }.get(mode, "System Online.")

            await session.send(input=greeting, end_of_turn=True)

            async def receive_from_gemini():
                try:
                    async for response in session.receive():
                        if response.text:
                            text_response = response.text
                            print(f"🤖 AI: {text_response}")

                            # Urgency Logic (Only for Safety Mode)
                            priority = "normal"
                            if mode == "safety" and "[CRITICAL]" in text_response:
                                priority = "high"

                            payload = {
                                "cmd": "speak", "text": text_response, "priority": priority}
                            await websocket.send_text(json.dumps(payload))
                except Exception:
                    pass

            async def receive_from_mobile():
                try:
                    while True:
                        data = await websocket.receive_text()
                        payload = json.loads(data)

                        # Handle Text & Images same as before...
                        if "text" in payload:
                            await session.send(input=payload["text"], end_of_turn=True)

                        if "image" in payload:
                            image_bytes = base64.b64decode(payload["image"])
                            await session.send(input={"mime_type": "image/jpeg", "data": image_bytes}, end_of_turn=True)
                except Exception:
                    pass

            await asyncio.gather(receive_from_gemini(), receive_from_mobile())

    except WebSocketDisconnect:
        print("❌ Disconnected")
    finally:
        try:
            await websocket.close()
        except:
            pass


# ... (Keep imports: json, asyncio, os, base64, traceback, fastapi, google.genai, dotenv, firebase_admin, credentials, auth) ...

# # --- BRAIN MODES (System Instructions) ---
# PERSONAS = {
#     "safety": """
#     You are Aura, a high-speed navigation guide for the blind.
#     **PRIORITY:** Safety & Orientation.
#     **RULES:**
#     1. **Clock Face:** "Door at 12 o'clock."
#     2. **Urgency:** Start with [CRITICAL] for dangers.
#     3. **Brevity:** Max 15 words. Imperative tone.
#     """,

#     "reading": """
#     You are Aura, a precise reading assistant.
#     **PRIORITY:** Optical Character Recognition (OCR).
#     **RULES:**
#     1. **Read Verbatim:** Read any visible text exactly as it appears.
#     2. **Context:** If text is cut off, say "Move camera right/left".
#     3. **Ignore Scenery:** Do not describe the table, hands, or background. Just the text.
#     """,

#     "scenery": """
#     You are Aura, a descriptive visual companion.
#     **PRIORITY:** Detail & Atmosphere.
#     **RULES:**
#     1. **Be Descriptive:** Describe colors, lighting, emotions, and aesthetics.
#     2. **Relaxed Tone:** Speak naturally and slowly. No urgency.
#     3. **Detail:** Mention textures, materials, and artistic details.
#     """
# }

# # ... (Keep verify_token and health_check same) ...

# @app.websocket("/ws")
# async def websocket_endpoint(websocket: WebSocket):
#     await websocket.accept()

#     # 1. Security Check
#     if not await verify_token(websocket):
#         await websocket.close(code=4001)
#         return

#     # 2. Determine Mode from URL (ws://...?mode=reading)
#     # Default to 'safety' if not specified
#     mode = websocket.query_params.get("mode", "safety")
#     selected_prompt = PERSONAS.get(mode, PERSONAS["safety"])

#     print(f"✅ Client Connected in [{mode.upper()}] Mode")

#     if not client:
#         await websocket.close(code=1011)
#         return

#     # 3. Configure Session with Selected Persona
#     session_config = {
#         "response_modalities": ["TEXT"],
#         "system_instruction": selected_prompt,
#     }

#     try:
#         async with client.aio.live.connect(model=MODEL, config=session_config) as session:
#             # Context-aware greeting
#             greeting = {
#                 "safety": "Safety Watch Active.",
#                 "reading": "Text Mode. Show me text.",
#                 "scenery": "Scenery Mode. Ready to describe."
#             }.get(mode, "System Online.")

#             await session.send(input=greeting, end_of_turn=True)

#             async def receive_from_gemini():
#                 try:
#                     async for response in session.receive():
#                         if response.text:
#                             text_response = response.text
#                             print(f"🤖 AI: {text_response}")

#                             # Urgency Logic (Only for Safety Mode)
#                             priority = "normal"
#                             if mode == "safety" and "[CRITICAL]" in text_response:
#                                 priority = "high"

#                             payload = {"cmd": "speak", "text": text_response, "priority": priority}
#                             await websocket.send_text(json.dumps(payload))
#                 except Exception:
#                     pass

#             async def receive_from_mobile():
#                 try:
#                     while True:
#                         data = await websocket.receive_text()
#                         payload = json.loads(data)

#                         # Handle Text & Images same as before...
#                         if "text" in payload:
#                             await session.send(input=payload["text"], end_of_turn=True)

#                         if "image" in payload:
#                             image_bytes = base64.b64decode(payload["image"])
#                             await session.send(input={"mime_type": "image/jpeg", "data": image_bytes}, end_of_turn=True)
#                 except Exception:
#                     pass

#             await asyncio.gather(receive_from_gemini(), receive_from_mobile())

#     except WebSocketDisconnect:
#         print("❌ Disconnected")
#     finally:
#         try:
#             await websocket.close()
#         except:
#             pass
