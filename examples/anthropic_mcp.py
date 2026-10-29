"""Claude + many MCP servers: list every server's tools, give Claude only the relevant ones.

pip install "laya-toolselect[semantic]" anthropic mcp
"""
import asyncio

from anthropic import Anthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from laya_toolselect import ToolSelector

SERVER = StdioServerParameters(command="npx", args=["-y", "@modelcontextprotocol/server-filesystem", "."])


async def main(request: str):
    async with stdio_client(SERVER) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            listed = await session.list_tools()          # ListToolsResult is accepted directly
            selector = ToolSelector(listed)
            picked = selector.select(request, k="auto")
            print("exposing", picked.names, f"({picked.savings:.0%} of tool text saved)")

            client = Anthropic()
            msg = client.messages.create(
                model="claude-sonnet-5-5", max_tokens=1024,
                tools=picked.as_anthropic(),                 # MCP inputSchema -> Anthropic input_schema
                messages=[{"role": "user", "content": request}])
            for block in msg.content:
                if block.type == "tool_use":
                    result = await session.call_tool(block.name, block.input)
                    print(block.name, "->", result.content[:1])


asyncio.run(main("list the markdown files in this folder"))
