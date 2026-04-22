import os

# https://bailian.console.aliyun.com/?tab=model#/api-key
os.environ["OPENAI_API_KEY"] = ""
os.environ["OPENAI_BASE_URL"] = "https://ark.cn-beijing.volces.com/api/v3"

import asyncio
import uuid

from openai.types.responses import ResponseContentPartDoneEvent, ResponseTextDeltaEvent
from agents import Agent, RawResponsesStreamEvent, Runner, TResponseInputItem, trace
# from agents.extensions.visualization import draw_graph
from agents import set_default_openai_api, set_tracing_disabled

set_default_openai_api("chat_completions")
set_tracing_disabled(True)

# 意图识别 -》 路由
# 用户提问 -》 类型1  类型2  类型3

math_agent = Agent(
    name="math_agent",
    model="doubao-seed-2-0-lite-260215",
    instructions="你是一个情感分析专家 李教授。你的任务是分析用户提供的文本，并判断其情感是“积极”、“消极”还是“中性”。请只返回这三个词中的一个作为答案。",
)

language_agent = Agent(
    name="language_agent",
    model="doubao-seed-2-0-lite-260215",
    instructions="你是一个命名实体识别（NER）专家 张教授。你的任务是从用户提供的文本中抽取出所有的人名、地名和组织机构名。如果找不到任何实体，请明确说明。请以清晰的列表形式返回结果。",
)

# triage 定义的的名字 默认的功能用户提问 指派其他agent进行完成
triage_agent = Agent(
    name="triage_agent",
    model="doubao-seed-2-0-lite-260215",
    instructions="你是一个任务分发总管。你的工作是分析用户的请求，并从你拥有的工具中选择最合适的一个来处理。你从不自己回答问题，总是使用你拥有的工具。",
    handoffs=[math_agent, language_agent],
)


async def main():
    # We'll create an ID for this conversation, so we can link each trace
    conversation_id = str(uuid.uuid4().hex[:16])

    try:
        draw_graph(triage_agent, filename="路由Handoffs")
    except:
        print("绘制agent失败，默认跳过。。。")

    msg = input("你好，我可以帮你处理文本的情感分类 或 文本的实体识别，请问你有什么问题？")
    agent = triage_agent
    inputs: list[TResponseInputItem] = [{"content": msg, "role": "user"}]

    while True:
        with trace("Routing example", group_id=conversation_id):
            result = Runner.run_streamed(
                agent,
                input=inputs,
            )
            async for event in result.stream_events():
                if not isinstance(event, RawResponsesStreamEvent):
                    continue
                data = event.data
                if isinstance(data, ResponseTextDeltaEvent):
                    print(data.delta, end="", flush=True)
                elif isinstance(data, ResponseContentPartDoneEvent):
                    print("\n")

        inputs = result.to_input_list()
        print("\n")

        user_msg = input("Enter a message: ")
        inputs.append({"content": user_msg, "role": "user"})
        agent = result.current_agent


if __name__ == "__main__":
    asyncio.run(main())
