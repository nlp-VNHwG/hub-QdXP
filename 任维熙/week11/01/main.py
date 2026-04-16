"""作业1：主 Agent 将用户请求交给子 Agent（情感分类 / 实体识别）处理。"""

from __future__ import annotations

import asyncio
import sys
import uuid
from pathlib import Path

_HOMEWORK_ROOT = Path(__file__).resolve().parent.parent
if str(_HOMEWORK_ROOT) not in sys.path:
    sys.path.insert(0, str(_HOMEWORK_ROOT))

from read_config import AppConfig, read_config

from agents import (
    Agent,
    HandoffOutputItem,
    ItemHelpers,
    MessageOutputItem,
    Runner,
    ToolCallItem,
    ToolCallOutputItem,
    TResponseInputItem,
    trace,
)
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX


def _build_model(cfg: AppConfig):
    from agents import AsyncOpenAI, OpenAIChatCompletionsModel
    from agents import set_default_openai_api, set_tracing_disabled

    set_default_openai_api("chat_completions")
    set_tracing_disabled(True)
    client = AsyncOpenAI(
        api_key=cfg.api_key,
        base_url=cfg.base_url.strip() or None,
    )
    return OpenAIChatCompletionsModel(model=cfg.model, openai_client=client)


_cfg = read_config()
_kw = {"model": _build_model(_cfg)}

sentiment_agent = Agent(
    name="情感分析专员",
    handoff_description="对用户提供的文本做情感分类（积极/消极/中性），并简要说明判断依据。",
    instructions=(
        f"{RECOMMENDED_PROMPT_PREFIX}\n"
        "你是情感分析专员。用户会提供一段中文或英文文本。\n"
        "请判断整体情感倾向：积极、消极或中性，并给出简短理由。\n"
        "不要编造文本中不存在的情节，仅基于给定内容分析。"
    ),
    **_kw,
)

ner_agent = Agent(
    name="实体识别专员",
    handoff_description="对文本进行命名实体识别，按类型列出人物、地点、机构、时间等。",
    instructions=(
        f"{RECOMMENDED_PROMPT_PREFIX}\n"
        "你是命名实体识别（NER）专员。用户会提供一段文本。\n"
        "请抽取并分类列出实体：人物、地点、组织机构、时间、其他专有名词。\n"
        "若某一类没有对应实体，明确写「无」。不要臆造未出现的实体。"
    ),
    **_kw,
)

triage_agent = Agent(
    name="主助手",
    handoff_description="接待用户，判断应交给情感分析还是实体识别。",
    instructions=(
        f"{RECOMMENDED_PROMPT_PREFIX}\n"
        "你是主调度助手。用户会描述想对一段文字做什么分析。\n"
        "规则：\n"
        "1）若用户关心情绪、褒贬、满意度、情感极性等，交给「情感分析专员」。\n"
        "2）若用户关心人名、地名、公司名、时间等实体抽取，交给「实体识别专员」。\n"
        "3）若同时需要两类分析，先交给情感分析专员；并告知用户如需实体列表可再发一条消息专门做 NER。\n"
        "请使用 handoff 将任务交给对应专员，不要自己直接完成分类或实体抽取。"
    ),
    handoffs=[sentiment_agent, ner_agent],
    **_kw,
)

sentiment_agent.handoffs.append(triage_agent)
ner_agent.handoffs.append(triage_agent)


async def main() -> None:
    current_agent: Agent = triage_agent
    input_items: list[TResponseInputItem] = []
    conversation_id = uuid.uuid4().hex[:16]

    print("多 Agent 路由已启动。输入 quit 退出。\n")

    while True:
        user_input = input("你: ").strip()
        if user_input.lower() in {"quit", "exit", "q"}:
            break
        if not user_input:
            continue

        with trace("Homework1 multi-agent router", group_id=conversation_id):
            input_items.append({"role": "user", "content": user_input})
            result = await Runner.run(current_agent, input_items)

            for new_item in result.new_items:
                agent_name = new_item.agent.name
                if isinstance(new_item, MessageOutputItem):
                    print(f"{agent_name}: {ItemHelpers.text_message_output(new_item)}")
                elif isinstance(new_item, HandoffOutputItem):
                    print(
                        f"[交接] {new_item.source_agent.name} → {new_item.target_agent.name}"
                    )
                elif isinstance(new_item, ToolCallItem):
                    print(f"{agent_name}: [调用工具]")
                elif isinstance(new_item, ToolCallOutputItem):
                    print(f"{agent_name}: [工具输出] {new_item.output}")
                else:
                    print(f"{agent_name}: [{new_item.__class__.__name__}]")

            input_items = result.to_input_list()
            current_agent = result.last_agent


if __name__ == "__main__":
    asyncio.run(main())

