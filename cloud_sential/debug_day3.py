import os
import asyncio
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from mcp.client.sse import sse_client
from mcp.client.session import ClientSession

# 1. Load Keys
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

print("🔍 --- DIAGNOSIS START ---")

# CHECK 1: The Keys
if not api_key:
    print("❌ FAILED: GOOGLE_API_KEY is missing from .env")
    exit(1)
else:
    print(f"✅ Key Found")

# CHECK 2: The Brain (Gemini)
print("\n🧠 Testing Gemini Connection...")
try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=api_key
    )
    # Simple "Hello World" to Google
    response = llm.invoke("Say 'Gemini is online'.")
    print(f"✅ Gemini Responded: {response.content}")
except Exception as e:
    print(f"❌ FAILED: Gemini Error: {e}")
    # If this fails, STOP. You can't run the agent without a brain.
    exit(1)

# CHECK 3: The Hands (MCP Server)
print("\n🔌 Testing MCP Connection (127.0.0.1)...")


async def test_mcp():
    url = "http://127.0.0.1:8001/sse"
    try:
        async with sse_client(url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                print(f"✅ MCP Connected! Found {len(tools.tools)} tools.")
                return True
    except Exception as e:
        print(f"❌ FAILED: MCP Error: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(test_mcp())
    print("\n-------------------------")
    print("If all checks passed, the Agent code is the issue.")
