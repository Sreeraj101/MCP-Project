import asyncio
import sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    # sys.executable ensures the subprocess runs inside the active .venv
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["server.py"],
        env=None
    )

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            # 1. Initialize the MCP handshake
            await session.initialize()
            print(" Connected to MCP Server successfully!\n")

            # 2. Discover available tools
            tools_response = await session.list_tools()
            print(" Discovered Tools on Server:")
            for tool in tools_response.tools:
                print(f" - {tool.name}: {tool.description}")

            # 3. Call the tool with arguments
            print("\n Calling 'fetch_github_repo_summary' for 'psf/requests'...\n")
            result = await session.call_tool(
                name="fetch_github_repo_summary",
                arguments={"owner": "psf", "repo": "requests"}
            )

            # 4. Print tool output
            for content in result.content:
                print("--- Result ---")
                print(content.text)

if __name__ == "__main__":
    asyncio.run(main())
    