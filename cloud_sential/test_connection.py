import asyncio
from mcp.client.sse import sse_client
from mcp.client.session import ClientSession


async def test_connection():
    print("🔌 Attempting to connect to MCP Server at http://localhost:8001/sse...")

    try:
        # This handles the tuple unpacking correctly
        async with sse_client("http://localhost:8001/sse") as (read_stream, write_stream):
            print("✅ SSE Connection Established!")

            async with ClientSession(read_stream, write_stream) as session:
                print("✅ Client Session Created!")

                await session.initialize()
                print("✅ Session Initialized!")

                tools = await session.list_tools()
                print(f"🎉 Success! Found {len(tools.tools)} tools:")
                for t in tools.tools:
                    print(f"   - {t.name}")

    except Exception as e:
        print(f"\n❌ CONNECTION FAILED: {e}")
        print("Check: Is python mcp-server/server.py running in another terminal?")

if __name__ == "__main__":
    asyncio.run(test_connection())
