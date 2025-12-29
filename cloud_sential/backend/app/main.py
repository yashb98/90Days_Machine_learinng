import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from pydantic import BaseModel, ConfigDict

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
cors_origins = os.getenv(
    "CORS_ORIGINS", "http://localhost:5173,http://localhost:3000,http://frontend:80").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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

    model_config = ConfigDict(extra="ignore")


class ChatRequest(BaseModel):
    message: str
    history: List[Message] = []

# --- ENDPOINTS ---


@app.get("/health")
async def health_check():
    """Health check endpoint for container orchestration"""
    return {"status": "healthy", "service": "cloud-sential-backend"}


@app.get("/policies", response_model=List[Policy])
async def get_policies(user_email: str = "unknown"):
    """
    Safely return policies list with defensive programming
    Ensures we always return a valid array to prevent frontend .map() errors
    """
    try:
        # Ensure UPLOADED_FILES_DB is always a list and contains valid data
        if not UPLOADED_FILES_DB:
            default_policy = {
                "id": "default",
                "name": "Default Security Standard (Built-in)",
                "status": "active",
                "lastUpdated": "System Boot"
            }
            return [default_policy]

        # Validate and clean the database
        cleaned_policies = []
        for policy in UPLOADED_FILES_DB:
            if isinstance(policy, dict) and all(key in policy for key in ["id", "name", "status", "lastUpdated"]):
                cleaned_policies.append(policy)

        # Always return at least the default policy
        if not cleaned_policies:
            default_policy = {
                "id": "default",
                "name": "Default Security Standard (Built-in)",
                "status": "active",
                "lastUpdated": "System Boot"
            }
            cleaned_policies.append(default_policy)

        return cleaned_policies

    except Exception as e:
        print(f"Error fetching policies: {e}")
        # Always return a valid array to prevent frontend crashes
        return [{
            "id": "error",
            "name": "Policy Service Unavailable",
            "status": "inactive",
            "lastUpdated": "Error State"
        }]


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
