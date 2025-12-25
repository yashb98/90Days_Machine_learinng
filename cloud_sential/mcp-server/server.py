from fastmcp import FastMCP
from tools.aws_audit import list_all_buckets, verify_s3_compliance

# 1. Initialize FastMCP (It handles FastAPI + SSE automatically)
mcp = FastMCP("CloudSentinel")

# 2. Register Tools using the decorator


@mcp.tool()
def list_s3_buckets() -> str:
    """
    Lists all S3 buckets available in the AWS account. 
    Use this first to find the correct bucket name.
    """
    return list_all_buckets()


@mcp.tool()
def audit_bucket_security(bucket_name: str) -> str:
    """
    Audits a specific S3 bucket for security compliance.
    Checks for: Server-Side Encryption (SSE), Versioning, and Public Access Blocks.
    """
    return verify_s3_compliance(bucket_name)


if __name__ == "__main__":
    print("🚀 CloudSentinel MCP Server starting on port 8001...")
    # Listen on 0.0.0.0 so the backend can reach it
    mcp.run(transport="sse", host="0.0.0.0", port=8001)
