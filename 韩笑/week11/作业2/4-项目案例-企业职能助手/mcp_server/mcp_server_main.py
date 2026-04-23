import asyncio
from fastmcp import FastMCP, Client

from homework import mcp as homework_mcp

mcp = FastMCP(
    name="MCP-Server"
)

async def setup():
    await mcp.import_server(homework_mcp, prefix="")

async def test_filtering():
    async with Client(mcp) as client:
        tools = await client.list_tools()
        print("Available tools:", [t.name for t in tools])


if __name__ == "__main__":
    asyncio.run(setup())
    asyncio.run(test_filtering())
    mcp.run(transport="sse", port=8900)
