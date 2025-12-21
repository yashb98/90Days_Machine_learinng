# A proper Websockt Manager that Handles JSON parsing and protocol logic

import json
from fastapi import WebSocket, WebSocketDisconnect


class WebSocketManager:
    def __init__(self):
        self.active_connections = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        # Default mode
        return "safety"

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def parse_message(self, raw_data: str):
        """
        Parses incoming JSON from Mobile.
        Expected format: 
        {
            "image": "base64...", 
            "text": "optional...", 
            "pose": {"lat": ..., "lng": ...} 
        }
        """
        try:
            return json.loads(raw_data)
        except json.JSONDecodeError:
            print("⚠️ Invalid JSON received")
            return {}

    async def send_response(self, websocket: WebSocket, text: str, priority: str = "normal"):
        response = {
            "type": "audio" if priority == "high" else "text",  # Simplified logic
            "content": text,
            "priority": priority
        }
        await websocket.send_json(response)
