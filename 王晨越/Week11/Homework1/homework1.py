import os

# https://bailian.console.aliyun.com/?tab=model#/api-key
os.environ["OPENAI_API_KEY"] = "sk-9f06aac1b31541958699954fe1ca8432"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

import asyncio
import uuid

from openai.types.responses import ResponseContentPartDoneEvent, ResponseTextDeltaEvent
from agents import Agent, RawResponsesStreamEvent, Runner, TResponseInputItem, trace
# from agents.extensions.visualization import draw_graph
from agents import set_default_openai_api, set_tracing_disabled
set_default_openai_api("chat_completions")
set_tracing_disabled(True)


sentiment_agent = Agent(
    name="Sentiment_Expert",
    instructions="You are a sentiment analysis expert. Classify text as Positive/Negative/Neutral.Always begin your response by stating who you are before answering the user's request.",
    model="qwen-max"
)

entity_agent = Agent(
    name="Entity_Expert",
    instructions="You are an Entity Classification Expert. State who you are, then extract specific entities like [Person], [Product], [Storage], [Location].",
    model="qwen-max"
)



# triage 定义的的名字 默认的功能用户提问 指派其他agent进行完成
triage_agent = Agent(
    name="triage_agent",
    model="qwen-max",
    instructions="""
    You are a strictly professional Router. You MUST NOT answer any user questions yourself.
    Follow these rules exactly:
    1. If the user mentions buying, products (iPhone, MacBook), people, or locations -> Transfer to Entity_Expert.
    2. If the user mentions feelings, mood, or emotions -> Transfer to Sentiment_Expert.
    3. If the request is unclear, ask for clarification.
    DO NOT perform the analysis yourself!
    """,
    handoffs=[entity_agent, sentiment_agent],
)


async def main():
    # We'll create an ID for this conversation, so we can link each trace
    conversation_id = str(uuid.uuid4().hex[:16])

    
    msg = input("Hello! I can help you with sentiment analysis and entity recognition. What else would you like to know? : ")
    agent = triage_agent
    inputs: list[TResponseInputItem] = [{"content": msg, "role": "user"}]

    while True:
        current_active_agent = triage_agent 
        with trace("Routing example", group_id=conversation_id):
            result = Runner.run_streamed(
                current_active_agent,
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