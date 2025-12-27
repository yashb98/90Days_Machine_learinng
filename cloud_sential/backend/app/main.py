import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from pydantic import BaseModel

# --- 1. RATE LIMITER IMPORTS ---
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# --- WORKER & AGENT IMPORTS ---
from backend.rag.ingest import process_document, UPLOADED_FILES_DB
from backend.app.agent import SecurityAgent

# --- 2. INITIALIZE LIMITER ---
# This identifies users by their IP address
limiter = Limiter(key_func=get_remote_address)

app = FastAPI()

# --- 3. REGISTER LIMITER TO APP ---
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# --- CORS MIDDLEWARE ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- DATA MODELS ---


class Policy(BaseModel):
    id: str
    name: str
    status: str
    lastUpdated: str


class Message(BaseModel):
    role: str
    content: str

    class Config:
        extra = "ignore"


class ChatRequest(BaseModel):
    message: str
    history: List[Message] = []

# --- ENDPOINTS ---


@app.get("/policies", response_model=List[Policy])
async def get_policies(user_email: str = "unknown"):
    if not UPLOADED_FILES_DB:
        return [{
            "id": "default",
            "name": "Default Security Standard (Built-in)",
            "status": "active",
            "lastUpdated": "System Boot"
        }]
    return UPLOADED_FILES_DB


@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    os.makedirs("temp_uploads", exist_ok=True)
    file_path = f"temp_uploads/{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        new_policy = await process_document(file_path, file.filename)
        return {"status": "success", "policy": new_policy}
    except Exception as e:
        print(f"Ingestion Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

# --- 4. APPLY LIMIT TO CHAT ---


@app.post("/chat")
@limiter.limit("5/minute")  # <--- User can only send 5 messages per minute
async def chat(request: Request, body: ChatRequest):
    # Note: We added 'request: Request' because slowapi needs it to check the IP

    agent = SecurityAgent()
    # We need to tell the agent about the history
    response_data = await agent.chat(body.message, body.history)
    return {"response": response_data}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
