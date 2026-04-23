import asyncio
import os

from agents import Agent, Runner, GuardrailFunctionOutput, InputGuardrail, set_default_openai_api, set_tracing_disabled
from pydantic import BaseModel

set_default_openai_api("chat_completions")
set_tracing_disabled(True)

# https://bailian.console.aliyun.com/?tab=model#/api-key
os.environ["OPENAI_API_KEY"] = "sk-f8cf10157f1d4271a502a38f2fffe040"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

class HomeworkOutput(BaseModel):
    """用于判断用户请求是否属于功课或学习类问题的结构"""
    is_homework: bool

guardrail_agent = Agent(
    name="Guardrail Check Agent",
    model="qwen-max",
    instructions="判断用户的问题是否属于情感分类、或实体识别问题。如果是，'is_homework'应为 True， json 返回",
    output_type=HomeworkOutput, # openai 官方推荐的一个语法，推荐大模型输出的格式类型，国内模型支持的不太好；
)

sentiment_classify_agent = Agent(
    name="sentiment classify expert",
    model='qwen-max',
    handoff_description='负责对文本进行情感分类的专家代理',
    instructions='你是专业的情感分析师，擅长对文本进行情感分类，输出文本的情感类型'
)

entity_recognizer_agent = Agent(
    name="entity recognizer expert",
    model='qwen-max',
    handoff_description='负责对文本进行实体识别的专家代理',
    instructions='你是专业的实体识别专家，擅长对文本进行实体识别，输出识别出来的所有实体'
)


async def homework_guardrail(ctx, agent, input_data):
    """
    运行检查代理来判断输入是否为功课。
    如果不是功课 ('is_homework' 为 False)，则触发阻断 (tripwire)。
    """
    print(f"\n[Guardrail Check] 正在检查输入: '{input_data}'...")

    # 运行检查代理
    result = await Runner.run(guardrail_agent, input_data, context=ctx.context)

    # 解析输出
    final_output = result.final_output_as(HomeworkOutput)

    tripwire_triggered = not final_output.is_homework

    return GuardrailFunctionOutput(
        output_info=final_output,
        tripwire_triggered=tripwire_triggered,
    )

route_agent = Agent(
    name="main agent",
    model='qwen-max',
    instructions="你的任务是根据用户输入的文本，判断应该将请求分派给'sentiment classify expert'还是'entity recognizer expert'",
    handoffs=[sentiment_classify_agent, entity_recognizer_agent],
    input_guardrails=[
        InputGuardrail(guardrail_function=homework_guardrail)
    ]
)

async def main():
    print("--- 启动中文代理系统示例 ---")
    try:
        query = '请对 明天我要带着小于去小芙家玩。 这句话做出实体识别'
        print(f"**用户提问:** {query}")
        result = await Runner.run(route_agent, query)
        print("\n**✅ 流程通过，最终输出:**")
        print(result.final_output)
    except Exception as e:
        print(e)

    try:
        query = '请对 我今天听了一个悲伤的故事，感同身受。 这句话做出情感分类'
        print(f"**用户提问:** {query}")
        result = await Runner.run(route_agent, query)
        print("\n**✅ 流程通过，最终输出:**")
        print(result.final_output)
    except Exception as e:
        print(e)

if __name__ == '__main__':
    asyncio.run(main())