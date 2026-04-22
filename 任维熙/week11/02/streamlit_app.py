"""
作业2：企业职能助手 Streamlit 前端。
先启动 MCP：在 mcp_server 目录执行  python mcp_server_main.py
再运行：在本目录执行  streamlit run streamlit_app.py

界面会展示：用户输入 → 模型选择的工具及参数 → 工具返回 → 模型总结。
"""

from __future__ import annotations

import asyncio
import sys
import traceback
from datetime import datetime
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="企业职能助手 · 作业")

_HOMEWORK_ROOT = Path(__file__).resolve().parent.parent
if str(_HOMEWORK_ROOT) not in sys.path:
    sys.path.insert(0, str(_HOMEWORK_ROOT))

try:
    from read_config import read_config

    cfg = read_config()
except Exception as e:
    st.error(f"配置加载失败: {e}")
    st.stop()

from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel, Runner, RunConfig, SQLiteSession
from agents import ModelSettings, set_default_openai_api, set_tracing_disabled
from agents.mcp import MCPServerSse
from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputItemDoneEvent,
    ResponseTextDeltaEvent,
)

set_default_openai_api("chat_completions")
set_tracing_disabled(True)

if "session" not in st.session_state:
    st.session_state.session = SQLiteSession("homework_enterprise_session")

ASSISTANT_INSTRUCTIONS = """
你是企业职能助手。可通过 MCP 调用多类工具。回答用户时要先判断意图，再选用工具。

【本次作业新增的三个自定义工具】（优先根据用户问题选用）：
1) query_department_budget_remaining — 查询部门年度剩余预算（参数 department_code，如 HR、RD、SALES、OPS）。
2) book_meeting_room — 预约会议室（参数 room_id, date_str=YYYY-MM-DD, start_hour, end_hour）。
3) generate_weekly_report_outline — 根据主题与要点生成周报 Markdown 大纲（参数 week_theme, highlights）。

还可使用天气、新闻、名言等其他已挂载工具。工具执行完成后，用中文简洁复述结果，不要编造工具未返回的数据。
""".strip()


with st.sidebar:
    st.title("选项")
    use_tool = st.checkbox("启用 MCP 工具", value=True)


def clear_history():
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "你好，我是企业职能助手（作业版）。勾选「启用 MCP 工具」后，我会按需调用工具并在此展示调用过程。",
        }
    ]
    st.session_state.session = SQLiteSession("homework_enterprise_session")


if "messages" not in st.session_state:
    clear_history()

st.sidebar.button("清空对话", on_click=clear_history)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


async def stream_with_tools(prompt: str, sqlite_session: SQLiteSession):
    external_client = AsyncOpenAI(
        api_key=cfg.api_key,
        base_url=cfg.base_url.strip() or None,
    )
    async with MCPServerSse(
        name="Homework SSE MCP",
        params={"url": "http://localhost:8900/sse"},
        cache_tools_list=False,
        client_session_timeout_seconds=30,
    ) as mcp_server:
        agent = Agent(
            name="EnterpriseAssistant",
            instructions=ASSISTANT_INSTRUCTIONS,
            mcp_servers=[mcp_server],
            model=OpenAIChatCompletionsModel(
                model=cfg.model,
                openai_client=external_client,
            ),
            model_settings=ModelSettings(parallel_tool_calls=False),
        )
        result = Runner.run_streamed(
            agent,
            input=prompt,
            session=sqlite_session,
            run_config=RunConfig(model_settings=ModelSettings(parallel_tool_calls=False)),
        )
        accumulated = ""
        async for event in result.stream_events():
            if event.type == "raw_response_event" and isinstance(
                event.data, ResponseOutputItemDoneEvent
            ):
                if isinstance(event.data.item, ResponseFunctionToolCall):
                    item = event.data.item
                    accumulated += (
                        f"\n\n**[工具选择]** `{item.name}`\n\n"
                        f"```json\n{item.arguments}\n```\n"
                    )
                    yield accumulated
            if event.type == "run_item_stream_event" and getattr(event, "name", "") == "tool_output":
                raw = event.item.raw_item
                if isinstance(raw, dict):
                    out = raw.get("output", "")
                else:
                    out = str(raw)
                accumulated += f"\n**[工具执行结果]**\n```text\n{out}\n```\n"
                yield accumulated
            if event.type == "raw_response_event" and isinstance(
                event.data, ResponseTextDeltaEvent
            ):
                accumulated += event.data.delta
                yield accumulated


if prompt := st.chat_input("请输入问题（例如：查一下 RD 部门还剩多少预算）"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        final_text = ""
        with st.spinner("思考与调用工具中…"):
            try:

                async def consume():
                    text = ""
                    sqlite_session = st.session_state.session
                    if use_tool:
                        agen = stream_with_tools(prompt, sqlite_session)
                        async for chunk in agen:
                            text = chunk
                            placeholder.markdown(text + "▌")
                    else:
                        external_client = AsyncOpenAI(
                            api_key=cfg.api_key,
                            base_url=cfg.base_url.strip() or None,
                        )
                        agent = Agent(
                            name="EnterpriseAssistant",
                            instructions="你是企业助手，不使用工具，直接回答。",
                            model=OpenAIChatCompletionsModel(
                                model=cfg.model,
                                openai_client=external_client,
                            ),
                        )
                        result = Runner.run_streamed(
                            agent, input=prompt, session=sqlite_session
                        )
                        async for event in result.stream_events():
                            if event.type == "raw_response_event" and isinstance(
                                event.data, ResponseTextDeltaEvent
                            ):
                                text += event.data.delta
                                placeholder.markdown(text + "▌")
                    return text

                final_text = asyncio.run(consume())
                placeholder.markdown(final_text)
            except Exception as e:
                final_text = f"发生错误: {e}\n\n```\n{traceback.format_exc()}\n```"
                placeholder.error(final_text)
                print(datetime.now(), e)
                traceback.print_exc()

    st.session_state.messages.append({"role": "assistant", "content": final_text})
