# Define what the API expects and returns.

from pydantic import BaseModel
from typing import List, Optional, Any


class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = "default"


class ToolLog(BaseModel):
    tool_name: str
    status: str
    args: Optional[dict] = None
    result: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    tool_logs: List[ToolLog] = []
