from backend.rag.services import GeminiEmbedderService, PineconeService
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import traceback
from dotenv import load_dotenv

# Load env vars immediately
load_dotenv()


# Import your RAG logic

# 1. Define the RAG Tool

@tool
def search_security_policy(query: str) -> str:
    """Search the security policy PDF for rules (e.g. 'encryption requirement')."""
    try:
        embedder = GeminiEmbedderService()
        vector_db = PineconeService(index_name="cloud-sentinel-gemini")
        query_vector = embedder.embed_query(query)
        results = vector_db.search(query_vector, top_k=3)
        if not results:
            return "No policy found."
        return "\n".join([doc.content for doc in results])
    except Exception as e:
        return f"RAG Error: {e}"

# 2. The Agent


class ComplianceAgent:
    def __init__(self):
        # Use Gemini Flash (Reliable & Fast)
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

    async def run_audit(self, user_query: str):
        messages = [
            SystemMessage(content="""You are CloudSentinel. 
1. Always search the policy FIRST using 'search_security_policy'.
2. Then use MCP tools to audit the actual AWS resources.
3. Compare the actual state vs the policy rules. Report VIOLATIONS strictly."""),
            HumanMessage(content=user_query)
        ]

        tool_logs = []

        try:
            print("Connecting to MCP Server at http://127.0.0.1:8001/sse ...")

            # Connect to the MCP Server
            async with sse_client("http://127.0.0.1:8001/sse") as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()

                    # --- EXPLICIT TOOL DEFINITIONS (STABLE) ---
                    # We define these wrappers here so they have access to 'session'

                    @tool
                    async def list_s3_buckets() -> str:
                        """Lists all S3 buckets available in the AWS account."""
                        result = await session.call_tool("list_s3_buckets")
                        return result.content[0].text

                    @tool
                    async def audit_bucket_security(bucket_name: str) -> str:
                        """Audits a specific S3 bucket for security compliance."""
                        # This explicit type hint (bucket_name: str) tells Gemini
                        # exactly what to pass.
                        result = await session.call_tool("audit_bucket_security", arguments={"bucket_name": bucket_name})
                        return result.content[0].text

                    # List of all tools available to the Brain
                    langchain_tools = [search_security_policy,
                                       list_s3_buckets, audit_bucket_security]

                    # Bind & Invoke
                    llm_with_tools = self.llm.bind_tools(langchain_tools)
                    ai_msg = await llm_with_tools.ainvoke(messages)
                    messages.append(ai_msg)

                    # Execute Tool Calls
                    if ai_msg.tool_calls:
                        for tc in ai_msg.tool_calls:
                            tool_logs.append(
                                {"tool": tc["name"], "args": tc["args"]})

                            # Find the matching tool
                            selected_tool = next(
                                t for t in langchain_tools if t.name == tc["name"])

                            # Run it
                            print(
                                f"🛠️ Agent calling tool: {tc['name']} with {tc['args']}")
                            if tc["name"] == "search_security_policy":
                                res = selected_tool.invoke(tc["args"])
                            else:
                                res = await selected_tool.ainvoke(tc["args"])

                            messages.append(ToolMessage(
                                tool_call_id=tc["id"], content=str(res)))

                        # Get Final Answer
                        final = await llm_with_tools.ainvoke(messages)
                        return final.content, tool_logs

                    return ai_msg.content, tool_logs

        except Exception as e:
            print("\nAGENT CRASHED ")
            traceback.print_exc()
            return f"Error running audit: {str(e)}", []
