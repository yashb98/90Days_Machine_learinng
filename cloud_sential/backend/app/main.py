from fastapi import FastAPI
from pydantic import BaseModel
from backend.app.agent import ComplianceAgent

app = FastAPI()


class Request(BaseModel):
    message: str


@app.post("/chat")
async def chat(req: Request):
    agent = ComplianceAgent()
    response, logs = await agent.run_audit(req.message)
    return {"response": response, "logs": logs}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
