import os
os.environ["OPENAI_API_KEY"] = "sk-4f0918ea000c45faa4274d3f171a8479"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

import asyncio
from openai.types.responses import ResponseContentPartDoneEvent, ResponseTextDeltaEvent
from agents import Agent, RawResponsesStreamEvent, Runner, TResponseInputItem, trace
#from agents.extensions.visualization import draw_graph
from agents import set_default_openai_api, set_tracing_disabled
set_default_openai_api("chat_completions")
set_tracing_disabled(True)

"""
    实现一个Agent做意图识别
    意图识别Agent可以分发handoff路由给两个子Agent
    两个子Agent分别是一个进行情感分类，一个做实体识别
    主Agent分发给两个子Agent，选择其中一个子Agent进行回答
"""

from agents import Agent, Runner, handoff, trace

sentiment_agent = Agent(
    name="sentiment_agent",
    model="qwen-max",
    instructions="""你是sentiment_agent，
    一个擅长做情感分析内容并且进行回答的专家，回答问题的时候告诉我你是谁"""
)

entity_agent = Agent(
    name="entity_agent",
    model="qwen-max",
    instructions="""你是entity_agent，
    一个擅长做实体识别内容并且进行回答的专家，回答问题的时候告诉我你是谁"""
)

# 主agent
triage_agent = Agent(
    name="intent_agent",
    model="qwen-max",
    instructions="""
                    你是triage_agent，
                    一个擅长意图识别内容并且进行回答的专家，
                    你需要做的事是把用户输入的问题交给sentiment_agent或entity_agent
    """,
    handoffs=[sentiment_agent, entity_agent],
)

async def main():
    #draw_graph(triage_agent, filename="意图识别")

    message = input("你好，我可以帮你回答情感类或者某个实体识别的任务")
    agent = triage_agent
    inputs = [{"content": message, "role": "user"}]
    while True:
        with trace("work1", group_id="work1"):
            result = Runner.run_streamed(agent, inputs)
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