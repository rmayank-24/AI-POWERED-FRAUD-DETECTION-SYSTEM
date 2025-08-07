import streamlit as st
import asyncio
import nest_asyncio
import json
import os
import platform

# --- ADDITION: Import both of your custom tools ---
from tools.fraud_tool import check_transaction_for_fraud
from tools.nlp_tool import analyze_text_for_risk
# ----------------------------------------------------

if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

nest_asyncio.apply()

if "event_loop" not in st.session_state:
    loop = asyncio.new_event_loop()
    st.session_state.event_loop = loop
    asyncio.set_event_loop(loop)

from langgraph.prebuilt import create_react_agent
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
from utils import astream_graph, random_uuid
from langchain_core.messages.ai import AIMessageChunk
from langchain_core.messages.tool import ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig

load_dotenv(override=True)

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

use_login = os.environ.get("USE_LOGIN", "false").lower() == "true"

if use_login and not st.session_state.authenticated:
    st.set_page_config(page_title="Intelligent Fraud Detection Agent", page_icon="🛡️")
else:
    st.set_page_config(page_title="Intelligent Fraud Detection Agent", page_icon="🛡️", layout="wide")

if use_login and not st.session_state.authenticated:
    st.title("🔐 Login")
    st.markdown("Login is required to use the system.")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit_button = st.form_submit_button("Login")
        if submit_button:
            expected_username = os.environ.get("USER_ID")
            expected_password = os.environ.get("USER_PASSWORD")
            if username == expected_username and password == expected_password:
                st.session_state.authenticated = True
                st.success("✅ Login successful! Please wait...")
                st.rerun()
            else:
                st.error("❌ Username or password is incorrect.")
    st.stop()

st.title("🛡️ Intelligent Fraud Detection Agent")
st.markdown("✨ Ask questions to the agent to analyze financial transactions.")

SYSTEM_PROMPT = """<ROLE>
You are a smart agent with an ability to use tools. 
You will be given a question and you will use the tools to answer the question.
Pick the most relevant tool to answer the question. 
If you are failed to answer the question, try different tools to get context.
Your answer should be very polite and professional.
</ROLE>
"""

OUTPUT_TOKEN_INFO = {
    "claude-3-5-sonnet-latest": {"max_tokens": 8192},
    "claude-3-5-haiku-latest": {"max_tokens": 8192},
    "claude-3-7-sonnet-latest": {"max_tokens": 64000},
    "gpt-4o": {"max_tokens": 16000},
    "gpt-4o-mini": {"max_tokens": 16000},
    "gemini-1.5-flash-latest": {"max_tokens": 8192},
    "llama3-8b-8192": {"max_tokens": 8192},
    "gemma-7b-it": {"max_tokens": 8192},
}

if "session_initialized" not in st.session_state:
    st.session_state.session_initialized = False
    st.session_state.agent = None
    st.session_state.history = []
    st.session_state.timeout_seconds = 120
    st.session_state.selected_model = "gemini-1.5-flash-latest"
    st.session_state.recursion_limit = 100

if "thread_id" not in st.session_state:
    st.session_state.thread_id = random_uuid()

def print_message():
    i = 0
    while i < len(st.session_state.history):
        message = st.session_state.history[i]
        if message["role"] == "user":
            st.chat_message("user", avatar="🧑‍💻").markdown(message["content"])
            i += 1
        elif message["role"] == "assistant":
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(message["content"])
                if (
                    i + 1 < len(st.session_state.history)
                    and st.session_state.history[i + 1]["role"] == "assistant_tool"
                ):
                    with st.expander("🔧 Tool Call Information", expanded=False):
                        st.markdown(st.session_state.history[i + 1]["content"])
                    i += 2
                else:
                    i += 1
        else:
            i += 1

def get_streaming_callback(text_placeholder, tool_placeholder):
    accumulated_text = []
    accumulated_tool = []

    def callback_func(message: dict):
        nonlocal accumulated_text, accumulated_tool
        message_content = message.get("content", None)
        if isinstance(message_content, AIMessageChunk):
            content = message_content.content
            if isinstance(content, list) and len(content) > 0:
                message_chunk = content[0]
                if message_chunk["type"] == "text":
                    accumulated_text.append(message_chunk["text"])
                    text_placeholder.markdown("".join(accumulated_text))
                elif message_chunk["type"] == "tool_use":
                    if "partial_json" in message_chunk:
                        accumulated_tool.append(message_chunk["partial_json"])
                    else:
                        tool_call_chunks = message_content.tool_call_chunks
                        tool_call_chunk = tool_call_chunks[0]
                        accumulated_tool.append(
                            "\n```json\n" + str(tool_call_chunk) + "\n```\n"
                        )
                    with tool_placeholder.expander("🔧 Tool Call Information", expanded=True):
                        st.markdown("".join(accumulated_tool))
            elif (
                hasattr(message_content, "tool_calls")
                and message_content.tool_calls
                and len(message_content.tool_calls[0]["name"]) > 0
            ):
                tool_call_info = message_content.tool_calls[0]
                accumulated_tool.append("\n```json\n" + str(tool_call_info) + "\n```\n")
                with tool_placeholder.expander("🔧 Tool Call Information", expanded=True):
                    st.markdown("".join(accumulated_tool))
            elif isinstance(content, str):
                accumulated_text.append(content)
                text_placeholder.markdown("".join(accumulated_text))
        elif isinstance(message_content, ToolMessage):
            accumulated_tool.append("\n```json\n" + str(message_content.content) + "\n```\n")
            with tool_placeholder.expander("🔧 Tool Call Information", expanded=True):
                st.markdown("".join(accumulated_tool))
        return None

    return callback_func, accumulated_text, accumulated_tool

async def process_query(query, text_placeholder, tool_placeholder, timeout_seconds=60):
    try:
        if st.session_state.agent:
            streaming_callback, accumulated_text_obj, accumulated_tool_obj = get_streaming_callback(text_placeholder, tool_placeholder)
            
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=query)
            ]

            try:
                response = await asyncio.wait_for(
                    astream_graph(
                        st.session_state.agent,
                        {"messages": messages},
                        callback=streaming_callback,
                        config=RunnableConfig(
                            recursion_limit=st.session_state.recursion_limit,
                            thread_id=st.session_state.thread_id,
                        ),
                    ),
                    timeout=timeout_seconds,
                )
            except asyncio.TimeoutError:
                error_msg = f"⏱️ Request time exceeded {timeout_seconds} seconds. Please try again later."
                return {"error": error_msg}, error_msg, ""

            final_text = "".join(accumulated_text_obj)
            final_tool = "".join(accumulated_tool_obj)
            return response, final_text, final_tool
        else:
            return {"error": "🚫 Agent has not been initialized."}, "🚫 Agent has not been initialized.", ""
    except Exception as e:
        import traceback
        error_msg = f"❌ Error occurred during query processing: {str(e)}\n{traceback.format_exc()}"
        return {"error": error_msg}, error_msg, ""

async def initialize_session():
    with st.spinner("🔄 Initializing agent..."):
        
        # --- FIX: Add the new NLP tool to the agent's tool list ---
        tools = [
            check_transaction_for_fraud, 
            analyze_text_for_risk
        ]
        # ---------------------------------------------------------
        
        st.session_state.tool_count = len(tools)
        
        selected_model = st.session_state.selected_model
        if "gemini" in selected_model:
            model = ChatGoogleGenerativeAI(model=selected_model, temperature=0.1)
        elif "llama" in selected_model or "gemma" in selected_model:
            model = ChatGroq(model_name=selected_model, temperature=0.1)
        elif "claude" in selected_model:
            model = ChatAnthropic(
                model=selected_model,
                temperature=0.1,
                max_tokens=OUTPUT_TOKEN_INFO.get(selected_model, {}).get("max_tokens", 4096),
            )
        else:
            model = ChatOpenAI(
                model=selected_model,
                temperature=0.1,
                max_tokens=OUTPUT_TOKEN_INFO.get(selected_model, {}).get("max_tokens", 4096),
            )
            
        agent = create_react_agent(model, tools, checkpointer=MemorySaver())
        
        st.session_state.agent = agent
        st.session_state.session_initialized = True
        return True

with st.sidebar:
    st.subheader("⚙️ System Settings")
    available_models = []
    has_groq_key = os.environ.get("GROQ_API_KEY") is not None
    if has_groq_key:
        available_models.extend(["llama3-8b-8192", "gemma-7b-it"])
    has_google_key = os.environ.get("GOOGLE_API_KEY") is not None
    if has_google_key:
        available_models.extend(["gemini-1.5-flash-latest"])
    has_anthropic_key = os.environ.get("ANTHROPIC_API_KEY") is not None
    if has_anthropic_key:
        available_models.extend(["claude-3-7-sonnet-latest", "claude-3-5-sonnet-latest", "claude-3-5-haiku-latest"])
    has_openai_key = os.environ.get("OPENAI_API_KEY") is not None
    if has_openai_key:
        available_models.extend(["gpt-4o", "gpt-4o-mini"])
    if not available_models:
        st.warning("⚠️ No API keys found. Please add GOOGLE_API_KEY, GROQ_API_KEY, ANTHROPIC_API_KEY, or OPENAI_API_KEY to your .env file.")
        available_models = ["gemini-1.5-flash-latest"]
    
    previous_model = st.session_state.selected_model
    st.session_state.selected_model = st.selectbox(
        "🤖 Select model to use",
        options=available_models,
        index=(available_models.index(st.session_state.selected_model) if st.session_state.selected_model in available_models else 0),
        help="Select a model. Free models require GOOGLE_API_KEY or GROQ_API_KEY.",
    )
    if previous_model != st.session_state.selected_model and st.session_state.session_initialized:
        st.warning("⚠️ Model has been changed. Click 'Apply Settings' button to apply changes.")
    
    st.session_state.timeout_seconds = st.slider("⏱️ Response generation time limit (seconds)", min_value=60, max_value=300, value=st.session_state.timeout_seconds, step=10)
    st.session_state.recursion_limit = st.slider("⏱️ Recursion call limit (count)", min_value=10, max_value=200, value=st.session_state.recursion_limit, step=10)
    st.divider()

with st.sidebar:
    st.subheader("📊 System Information")
    st.write(f"🛠️ Tools Available: {st.session_state.get('tool_count', 'Initializing...')}")
    selected_model_name = st.session_state.selected_model
    st.write(f"🧠 Current Model: {selected_model_name}")
    if st.button("Apply Settings", key="apply_button", type="primary", use_container_width=True):
        apply_status = st.empty()
        with apply_status.container():
            st.warning("🔄 Applying changes. Please wait...")
            progress_bar = st.progress(0)
            st.session_state.session_initialized = False
            st.session_state.agent = None
            progress_bar.progress(30)
            success = st.session_state.event_loop.run_until_complete(initialize_session())
            progress_bar.progress(100)
            if success:
                st.success("✅ New settings have been applied.")
            else:
                st.error("❌ Failed to apply settings.")
        st.rerun()
    st.divider()
    st.subheader("🔄 Actions")
    if st.button("Reset Conversation", use_container_width=True, type="primary"):
        st.session_state.thread_id = random_uuid()
        st.session_state.history = []
        st.success("✅ Conversation has been reset.")
        st.rerun()
    if use_login and st.session_state.authenticated:
        st.divider()
        if st.button("Logout", use_container_width=True, type="secondary"):
            st.session_state.authenticated = False
            st.success("✅ You have been logged out.")
            st.rerun()

if not st.session_state.session_initialized:
    st.info("Agent is not initialized. Please click the 'Apply Settings' button in the left sidebar to initialize.")

print_message()

user_query = st.chat_input("💬 Enter your question")
if user_query:
    if st.session_state.session_initialized:
        st.chat_message("user", avatar="🧑‍💻").markdown(user_query)
        with st.chat_message("assistant", avatar="🤖"):
            tool_placeholder = st.empty()
            text_placeholder = st.empty()
            resp, final_text, final_tool = st.session_state.event_loop.run_until_complete(
                process_query(user_query, text_placeholder, tool_placeholder, st.session_state.timeout_seconds)
            )
        if "error" in resp:
            st.error(resp["error"])
        else:
            st.session_state.history.append({"role": "user", "content": user_query})
            st.session_state.history.append({"role": "assistant", "content": final_text})
            if final_tool.strip():
                st.session_state.history.append({"role": "assistant_tool", "content": final_tool})
            st.rerun()
    else:
        st.warning("⚠️ Agent is not initialized. Please click the 'Apply Settings' button in the left sidebar to initialize.")
