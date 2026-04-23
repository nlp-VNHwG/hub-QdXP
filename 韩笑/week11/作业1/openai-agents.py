import os

os.environ["OPENAI_API_KEY"] = "sk-777ae59d8b3e451db4dd91fe6961dbe5"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

import asyncio
from pydantic import BaseModel
from agents import Agent, InputGuardrail, GuardrailFunctionOutput, Runner
from agents.exceptions import InputGuardrailTripwireTriggered
from agents import set_default_openai_api, set_tracing_disabled

set_default_openai_api("chat_completions")
set_tracing_disabled(True)


class ClassificationOutput(BaseModel):
    is_agent: bool


guardrail_agent = Agent(
    name="guardrail",
    model="qwen-max",
    instructions="判断用户请求输入是否为大模型学习相关问题，如情感分类，实体识别等，如果是'is_agent'应为True，返回值类型为json格式",
    output_type=ClassificationOutput
)

emotion_agent = Agent(
    name="emotion",
    model="qwen-max",
    handoff_description="负责进行情感分类的专家",
    instructions="你是情感分类的专家，可以根据用户的输入进行准确的情感分类，并只会返回对应的情感类型"
)

entity_agent = Agent(
    name="entity",
    model="qwen-max",
    handoff_description="负责进行实体识别的专家",
    instructions="你是实体识别的专家，可以将用户的需要进行实体识别句子，进行准确的实体识别，并只会返回识别完成后的数据"
)


async def classification_guardrail(ctx, agent, input_data):
    print(f"f\n[guardrail] 正在检查输入:'{input_data}'....")

    # 运行检查
    result = await Runner.run(guardrail_agent, input_data, context=ctx.context)

    # 解析输出
    final_output = result.final_output_as(ClassificationOutput)

    tripwire_triggered = not final_output.is_agent

    return GuardrailFunctionOutput(
        output_info=final_output,
        tripwire_triggered=tripwire_triggered
    )


triage_agent = Agent(
    name="triage",
    model="qwen-max",
    instructions="你的任务是根据用户的问题进行请求判断，分给'emotion_agent' 或者 'entity_agent' ",
    handoffs=[entity_agent, emotion_agent],
    input_guardrails=[
        InputGuardrail(guardrail_function=classification_guardrail),
    ]
)

async def main():
    print("启动openai-agent实例服务")

    try:
        query = "判断下面这句话的情绪：我今天中了五百万，我很开心"
        print(f"用户提问:{query}'")
        result = await Runner.run(triage_agent, query)
        print("流程通过")
        print(result.final_output)
    except InputGuardrailTripwireTriggered as e:
        print("\n**❌ 守卫阻断触发:**", e)

    print("\n" + "=" * 50)
    print("=" * 50)

    try:
        query = "对下面这句话进行实体识别：我今天中了五百万，我很开心"
        print(f"用户提问:{query}'")
        result = await Runner.run(triage_agent, query)
        print("流程通过")
        print(result.final_output)
    except InputGuardrailTripwireTriggered as e:
        print("\n**❌ 守卫阻断触发:**", e)

    print("\n" + "=" * 50)
    print("=" * 50)

    try:
        query = "你好，你的功能是什么"
        print(f"用户提问:{query}'")
        result = await Runner.run(triage_agent, query)
        print("流程通过")
        print(result.final_output)
    except InputGuardrailTripwireTriggered as e:
        print("\n**❌ 守卫阻断触发:**", e)

if __name__ == "__main__":
    asyncio.run(main())