import json
import asyncio
import base64
import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

app = FastAPI()


@app.get("/health")
async def health_check():
    return {
        "status": "online",
        "message": "Aura Backend is running!"
    }


# 1. Connection Manager to handle multiple users (Scaling prep)


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print("Client Connected")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        print("Client Disconnected")


manager = ConnectionManager()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # 2. Receive Data (JSON) from Flutter
            # Expected Format: {"image": "base64_string...", "mime_type": "image/jpeg"}
            data = await websocket.receive_text()
            payload = json.loads(data)

            # 3. Process the "Eyes" (Log the size to prove we got it)
            image_chunk = payload.get("image", "")
            print(f" Received Frame: {len(image_chunk) // 1024} KB")

            # 4. Simulate the "Voice" (Response)
            # In Day 51, this will be replaced by real Gemini Audio bytes
            # For now, we echo back a confirmation to the phone
            response = {
                "status": "processed",
                "message": "Frame received by Brain"
            }
            await websocket.send_text(json.dumps(response))

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"Error: {e}")
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.close()
