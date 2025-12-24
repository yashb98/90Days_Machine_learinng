# MCP server Entry point
import uvicorn
import os
from fastapi import FastAPI
from mcp.server.fastapi import McpServer
from mcp.server.sse import SseServerTransport
from dotenv import load_dotenv

# Import our actual tools
from tools.aws_audit import list_all_buckets, verify_s3_compliance

# Load env vars
load_dotenv()

# 1. Initialize MCP Server
mcp = McpServer(name="CloudSentinel")

# 2. Register Tools
# The docstring is CRITICAL. It tells the LLM *when* to use this tool.


@mcp.tool()
async def list_s3_buckets() -> str:
    """
    Lists all S3 buckets available in the AWS account. 
    Use this first to find the correct bucket name if the user doesn't provide it.
    """
    return list_all_buckets()


@mcp.tool()
async def audit_bucket_security(bucket_name: str) -> str:
    """
    Audits a specific S3 bucket for security compliance.
    Checks for: Server-Side Encryption (SSE), Versioning, and Public Access Blocks.

    Args:
        bucket_name: The exact name of the S3 bucket to inspect.
    """
    return verify_s3_compliance(bucket_name)

# 3. Create FastAPI Wrapper (for SSE transport)
app = FastAPI()

# 4. Expose MCP via SSE
# The Client (Agent) will connect to /sse to receive instructions
mcp_transport = SseServerTransport("/sse")
app.include_router(mcp.router(mcp_transport))

if __name__ == "__main__":
    print("🚀 CloudSentinel MCP Server starting on port 8001...")
    # Port 8001 matches our docker-compose plan
    uvicorn.run(app, host="0.0.0.0", port=8001)
