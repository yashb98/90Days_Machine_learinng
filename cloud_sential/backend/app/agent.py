import os
import traceback
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage,  ToolMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# MCP Client imports
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client

# RAG Imports (Ensure these paths match your project structure)
from backend.rag.services import GeminiEmbedderService, PineconeService

# Load env vars
load_dotenv()

# --- 1. Define the RAG Tool ---


@tool
def search_security_policy(query: str) -> str:
    """Search the security policy PDF for rules."""
    print(f"\n🔍 DEBUG: Searching Policy for: '{query}'")

    try:
        embedder = GeminiEmbedderService()
        query_vector = embedder.embed_query(query)
        vector_db = PineconeService()

        # Check if index even has vectors (Handling "Fresh Boot" error)
        stats = vector_db.index.describe_index_stats()
        if stats['total_vector_count'] == 0:
            return "SYSTEM_ALERT: The Knowledge Base is empty. Please upload a policy PDF first."

        results = vector_db.search(query_vector, top_k=3)

        if not results:
            return "No specific policy details found for this query."

        # 3. Format Results (Robust Fix)
        formatted_results = []
        for doc in results:
            # Check which attribute holds the text
            if hasattr(doc, 'page_content'):
                text = doc.page_content
            elif hasattr(doc, 'content'):
                text = doc.content
            else:
                # Fallback: Try to find text in dictionary or string
                text = str(doc)

            formatted_results.append(text)

        return "\n".join(formatted_results)

    except Exception as e:
        print("\n❌ RAG ERROR:")
        traceback.print_exc()
        return f"RAG Error: {e}"


# --- 2. The Agent Class ---
class SecurityAgent:
    def __init__(self):
        # Use Gemini Flash (Reliable & Fast)
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

    # 1. Update signature to accept 'history'
    async def chat(self, user_query: str, history: list = []):

        # 2. Start with the System Prompt
        messages = [
            SystemMessage(content="""You are CloudSentinel. 
1. Always search the policy FIRST using 'search_security_policy'.
2. Then use MCP tools to audit the actual AWS resources.
3. Compare the actual state vs the policy rules. Report VIOLATIONS strictly.""")
        ]

        # 3. CONVERT & APPEND HISTORY
        # We loop through the history provided by the frontend and convert it
        # into the format LangChain understands.
        for msg in history:
            if msg.role == "user":
                messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant" or msg.role == "model":
                messages.append(AIMessage(content=msg.content))

        # 4. Add the NEW User Query at the end
        messages.append(HumanMessage(content=user_query))

        tool_logs = []

        try:
            mcp_server_url = os.getenv(
                "MCP_SERVER_URL", "http://mcp-server:8001/sse")
            print(f"🔌 Connecting to MCP Server at {mcp_server_url} ...")

            # Connect to the MCP Server
            async with sse_client(mcp_server_url) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()

                    # --- Define MCP Tools Wrappers ---
                    @tool
                    async def list_s3_buckets() -> str:
                        """Lists all S3 buckets available in the AWS account."""
                        result = await session.call_tool("list_s3_buckets")
                        return result.content[0].text

                    @tool
                    async def audit_bucket_security(bucket_name: str) -> str:
                        """Audits a specific S3 bucket for security compliance."""
                        result = await session.call_tool("audit_bucket_security", arguments={"bucket_name": bucket_name})
                        return result.content[0].text

                    # Bind Tools
                    langchain_tools = [search_security_policy,
                                       list_s3_buckets, audit_bucket_security]
                    llm_with_tools = self.llm.bind_tools(langchain_tools)

                    # 5. First Thought (AI decides which tool to use)
                    # We pass the full 'messages' list (System + History + Current Query)
                    ai_msg = await llm_with_tools.ainvoke(messages)
                    messages.append(ai_msg)

                    # 6. Execute Tools (if any)
                    if ai_msg.tool_calls:
                        for tc in ai_msg.tool_calls:
                            # Log for Frontend
                            tool_logs.append(
                                {"tool": tc["name"], "args": tc["args"]})
                            print(
                                f"🛠️ Agent calling tool: {tc['name']} with {tc['args']}")

                            # Find and Run Tool
                            selected_tool = next(
                                t for t in langchain_tools if t.name == tc["name"])

                            if tc["name"] == "search_security_policy":
                                res = selected_tool.invoke(
                                    tc["args"])  # Sync tool
                            else:
                                # Async MCP tool
                                res = await selected_tool.ainvoke(tc["args"])

                            messages.append(ToolMessage(
                                tool_call_id=tc["id"], content=str(res)))

                        # 7. Final Answer (after seeing tool results)
                        final = await llm_with_tools.ainvoke(messages)

                        return {
                            "response": final.content,
                            "logs": tool_logs
                        }

                    # If no tools were called, return the direct response
                    return {
                        "response": ai_msg.content,
                        "logs": tool_logs
                    }

        except Exception as e:
            print("\n❌ AGENT CRASHED")
            traceback.print_exc()
            return {
                "response": f"System Error: {str(e)}",
                "logs": []
            }
