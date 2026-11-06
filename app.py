import asyncio
import os
import sys
from pathlib import Path
import streamlit as st
from google import genai
from google.genai import types
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

st.set_page_config(
    page_title="MCP DevOps Assistant - Live Demo",
    page_icon="🛠️",
    layout="centered"
)

st.title("🛠️ DevOps Assistant (FastMCP Agent)")
st.caption(
    "Live demonstration of an autonomous Model Context Protocol (MCP) tool server running "
    "over stdio, integrated with Google Gemini function calling."
)

# 1. API Key resolution (reads from Streamlit Secrets or manual input)
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]

if not api_key:
    with st.sidebar:
        st.header("Authentication")
        api_key = st.text_input("Enter Gemini API Key", type="password")
        st.info("Tip: Add GEMINI_API_KEY to Streamlit Secrets so visitors don't need a key.")

# 2. Agent Execution Loop
async def run_mcp_agent(query: str, key: str):
    current_dir = Path(__file__).parent.resolve()
    server_script = str(current_dir / "server.py")

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[server_script],
        env=dict(os.environ),
    )

    client = genai.Client(api_key=key)

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # Dynamic tool discovery via MCP protocol
            mcp_tools = await session.list_tools()
            function_declarations = [
                {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.inputSchema,
                }
                for t in mcp_tools.tools
            ]

            tools = [{"function_declarations": function_declarations}]

            chat = client.aio.chats.create(
                model="gemini-2.5-flash",
                config=types.GenerateContentConfig(
                    tools=tools,
                    temperature=0.0
                )
            )

            # Turn 1: Model decides whether to call the tool
            response = await chat.send_message(query)

            tool_data_for_ui = None
            if response.function_calls:
                call = response.function_calls[0]
                tool_name = call.name
                tool_args = dict(call.args)

                # Execute tool via MCP over stdio
                try:
                    tool_result = await session.call_tool(
                        name=tool_name,
                        arguments=tool_args
                    )
                    tool_output = "\n".join([c.text for c in tool_result.content])
                except Exception as e:
                    tool_output = f"Tool execution failed: {str(e)}"

                tool_data_for_ui = {
                    "tool_name": tool_name,
                    "arguments": tool_args,
                    "output": tool_output
                }

                # Turn 2: Feed output back to model for synthesis
                final_response = await chat.send_message(
                    types.Part.from_function_response(
                        name=tool_name,
                        response={"result": tool_output}
                    )
                )
                return tool_data_for_ui, final_response.text
            else:
                return None, response.text

# 3. Interactive UI
default_prompts = [
    "Can you check the GitHub stats for psf/requests and summarize its community health?",
    "Check repo metrics for pallets/flask.",
    "Can you check the GitHub stats for non_existent_user_9999/fake_repo?"
]

selected_prompt = st.selectbox("Sample prompts to test:", default_prompts)
custom_query = st.text_input("Or enter a custom prompt:", value=selected_prompt)

if st.button("Run Agent Query", type="primary"):
    if not api_key:
        st.error("Please provide a Gemini API Key to run the agent.")
    else:
        with st.spinner("Connecting to MCP server via stdio and querying Gemini..."):
            try:
                tool_event, final_text = asyncio.run(run_mcp_agent(custom_query, api_key))

                # Display tool execution breakdown
                if tool_event:
                    with st.expander("🔍 MCP Tool Handshake & Invocation", expanded=True):
                        st.markdown(f"**Tool Called:** `{tool_event['tool_name']}`")
                        st.json(tool_event["arguments"])
                        st.markdown("**Structured Response from Server:**")
                        st.code(tool_event["output"], language="text")

                st.subheader("Agent Answer")
                st.markdown(final_text)

            except Exception as ex:
                st.error(f"Execution Error: {str(ex)}")