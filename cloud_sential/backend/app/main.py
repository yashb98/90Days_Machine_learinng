from fastapi import FastAPI
from pydantic import BaseModel
from backend.app.agent import ComplianceAgent
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

app = FastAPI()
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


class Request(BaseModel):
    message: str


@app.post("/chat")
@limiter.limit("5/minute")
async def chat(req: Request):
    agent = ComplianceAgent()
    response, logs = await agent.run_audit(req.message)
    return {"response": response, "logs": logs}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
