import asyncio
import os
import sys
from pathlib import Path
from google import genai
from google.genai import types
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AQ.*********************")
MODEL_ID = "gemini-flash-lite-latest"

async def run_agent(user_query: str):
    current_dir = Path(__file__).parent.resolve()
    server_script = str(current_dir / "server.py")

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[server_script],
        env=dict(os.environ),
    )

    client = genai.Client(api_key=GEMINI_API_KEY)

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            print("Connected to MCP Server.")

            # 1. Discover tools dynamically from MCP server
            mcp_tools = await session.list_tools()

            # 2. Map MCP tools to Gemini function declarations
            function_declarations = []
            for t in mcp_tools.tools:
                function_declarations.append({
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.inputSchema
                })

            tools = [{"function_declarations": function_declarations}]

            # 3. Create a chat session with the tools registered
            chat = client.aio.chats.create(
                model=MODEL_ID,
                config=types.GenerateContentConfig(
                    tools=tools,
                    temperature=0.0
                )
            )

            print(f"\nUser Query: {user_query}")
            print(f"Asking Gemini ({MODEL_ID})...")

            # 4. Turn 1: Model determines if a tool call is needed
            response = await chat.send_message(user_query)

            # 5. Check for function calls
            if response.function_calls:
                call = response.function_calls[0]
                tool_name = call.name
                tool_args = dict(call.args)
                print(f"\nLLM invoked tool: '{tool_name}' with arguments: {tool_args}")

                # Call tool via MCP server over stdio
                try:
                    tool_result = await session.call_tool(
                        name=tool_name,
                        arguments=tool_args
                    )
                    tool_output = "\n".join([c.text for c in tool_result.content])
                except Exception as e:
                    tool_output = f"Tool execution failed: {str(e)}"

                print(f"\nTool Output from MCP:\n{tool_output}\n")

                # 6. Turn 2: Feed the tool result back into the chat session
                print("Synthesizing final answer with Gemini...")
                final_response = await chat.send_message(
                    types.Part.from_function_response(
                        name=tool_name,
                        response={"result": tool_output}
                    )
                )
                print(f"\nFinal Agent Answer:\n{final_response.text}")
            else:
                print("\nDirect Answer (no tool needed):", response.text)

if __name__ == "__main__":
    test_prompt = "Can you check the GitHub stats for non_existent_user_9999/fake_repo?"
    asyncio.run(run_agent(test_prompt))