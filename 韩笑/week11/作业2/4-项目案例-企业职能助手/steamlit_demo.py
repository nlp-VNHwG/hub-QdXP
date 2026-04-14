import traceback
from datetime import datetime

import streamlit as st
from agents.mcp.server import MCPServerSse
import asyncio
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, SQLiteSession, RunConfig, ModelSettings
from openai.types.responses import ResponseTextDeltaEvent, ResponseCreatedEvent, ResponseOutputItemDoneEvent, \
    ResponseFunctionToolCall
from agents.mcp import MCPServer, ToolFilterStatic, ToolFilterCallable
from agents import set_default_openai_api, set_tracing_disabled

# OpenAI-agent settings
set_default_openai_api("chat_completions")
set_tracing_disabled(True)

st.set_page_config(page_title="企业职能机器人")

# SQLite session for conversation
session = SQLiteSession("conversation_123")

# Sidebar
with st.sidebar:
    st.title('职能AI+智能问答')
    if 'API_TOKEN' in st.session_state and len(st.session_state['API_TOKEN']) > 1:
        st.success('API Token已经配置', icon='✅')
        key = st.session_state['API_TOKEN']
    else:
        key = "sk-399b434c3f5b4329a4600ec76ce4f7cc"

    key = st.text_input('输入API KEY:', type='password', value=key)
    st.session_state['API_TOKEN'] = key # 网页缓存

    model_name = st.selectbox("选择模型", ["qwen-flash", "qwen-max"])
    use_tool = st.checkbox("使用工具")


# Initial chat messages
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "你好，我是企业职能助手，可以AI对话 也 可以调用内部工具。"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]): # 对话角色
        st.write(message["content"]) # 对外输出


def clear_chat_history():
    st.session_state.messages = [
        {"role": "system", "content": "你好，我是企业职能助手，可以AI对话 也 可以调用内部工具。"}
    ]
    global session
    session = SQLiteSession("conversation_123")


st.sidebar.button('清空聊天', on_click=clear_chat_history)


def mcp_homework_callable_filter(context, tool):
    return tool.name == "get_exercises_advice" or tool.name == "get_english_homework" or tool.name == "get_math_homework"

# 动态的工具的选择
async def get_model_response3(prompt, model_name, use_tool):


    mcp_server3 = MCPServerSse(
        name="SSE Python Server",
        params={"url": "http://localhost:8900/sse"},
        cache_tools_list=False,
        tool_filter=mcp_homework_callable_filter,
        client_session_timeout_seconds=20,
    )


    external_client = AsyncOpenAI(
        api_key=key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    async with  mcp_server3:

        if use_tool:

            homework_agent = Agent(
                name="Homework Assistant",
                instructions=(
                    "You are an expert in fitness and health and homework. "
                    "You can calculate body fat percentage and provide exercise advice. "
                    "If the user provides age, weight, height, and sex, use the get_exercises_advice tool. "
                    "If the user asks for general fitness tips, try to answer based on your knowledge."
                    "If the user asks for english homework tips"
                    "If the user asks for math homework tips"
                ),
                mcp_servers=[mcp_server3],
                model=OpenAIChatCompletionsModel(
                    model=model_name,
                    openai_client=external_client,
                ),
                model_settings=ModelSettings(parallel_tool_calls=False)
            )
            agent = Agent(
                name="triage_agent",
                instructions=(
                    "Route the user's request to the appropriate specialized agent based on the task intent. "
                    "Do not try to answer the question yourself, just route it. "
                    "If the user asks about weather, sentiment, or general tools, use the Tool Assistant. "
                    "If the user asks about exercise advice, english homework, or math homework, use the Homework Assistant. "
                    "If the user asks about news, use the News Assistant."
                ),
                handoffs=[homework_agent],
                model=OpenAIChatCompletionsModel(
                    model=model_name,
                    openai_client=external_client,
                ),
                model_settings=ModelSettings(parallel_tool_calls=False)
            )
        else:
            agent = Agent(
                name="Assistant",
                instructions="",
                model=OpenAIChatCompletionsModel(
                    model=model_name,
                    openai_client=external_client,
                )
            )

        result = Runner.run_streamed(agent, input=prompt, session=session, run_config=RunConfig(model_settings=ModelSettings(parallel_tool_calls=False)))

        async for event in result.stream_events():
            print(datetime.now(), "111", event)

            if event.type == "raw_response_event" and hasattr(event, 'data') and isinstance(event.data, ResponseOutputItemDoneEvent):
                print(datetime.now(), "222", event)
                if isinstance(event.data.item, ResponseFunctionToolCall):
                    yield "argument", event.data.item

            if event.type == "run_item_stream_event" and hasattr(event, 'name') and event.name == "tool_output":
                print(datetime.now(), "333", event)
                yield "raw", event.item.raw_item["output"]

            if event.type == "raw_response_event" and hasattr(event, 'data') and isinstance(event.data, ResponseTextDeltaEvent):
                print(datetime.now(), "444", event)
                yield "content", event.data.delta


# ----------------------------
#    Chat Interaction
# ----------------------------
if len(key) > 1:
    if prompt := st.chat_input(): # 得到用户输入，判断输入是否为空
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})

        # 用户的角色暂时用户的输入
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant streaming reply
        with st.chat_message("assistant"):
            placeholder = st.empty()

            with st.spinner("请求中..."):
                try:
                    # 把 streaming consumer 放成一个独立的 async 函数（使用局部 accumulated_text）
                    async def stream_output():
                        accumulated_text = ""

                        # 生成结果的迭代器
                        response_generator = get_model_response3(prompt, model_name, use_tool)

                        async for event_type, chunk in response_generator:

                            # Raw event（原始 delta），把它格式化为 code block，方便查看
                            if event_type == "argument":
                                # chunk 可能是 dict/list/其他对象 —— 转成字符串以防报错
                                formatted_raw = f"\n\n```json\n[RawArg]\n{str(chunk)}\n```\n"
                                accumulated_text += formatted_raw
                                placeholder.markdown(accumulated_text + "▌")

                            elif event_type == "raw":
                                # chunk 可能是 dict/list/其他对象 —— 转成字符串以防报错
                                formatted_raw = f"\n\n```json\n[RawEvent]\n{str(chunk)}\n```\n"
                                accumulated_text += formatted_raw
                                placeholder.markdown(accumulated_text + "▌")

                            # 模型输出文本
                            elif event_type == "content":
                                # chunk 应该是 str（文本片段）
                                accumulated_text += chunk
                                placeholder.markdown(accumulated_text + "▌")

                        return accumulated_text

                    # 在同步上下文中运行 async generator
                    final_text = asyncio.run(stream_output())
                    # 最终渲染（去掉游标）
                    placeholder.markdown(final_text)

                except Exception as e:
                    error_msg = f"发生错误: {e}"
                    placeholder.error(error_msg)
                    final_text = error_msg
                    traceback.print_exc()

            # Save assistant reply to session state
            st.session_state.messages.append({"role": "assistant", "content": final_text})
