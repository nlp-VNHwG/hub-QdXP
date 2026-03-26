import os
import json
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import openai

dotenv_path = os.path.join(os.path.dirname(__file__), '..', 'env', '.env')
load_dotenv(dotenv_path = dotenv_path)

# 初始化智谱AI客户端
# 如果没有设置环境变量，可以手动替换为你的API Key
client = client = openai.OpenAI(
    api_key=os.getenv("DOUBAO_API_KEY"),
    base_url="https://ark.cn-beijing.volces.com/api/v3",
)
# --- 1. 定义你的“工具” ---
# 我们使用 Pydantic 来定义一个结构清晰的“翻译”工具。
# LLM 会理解这个类的描述和字段，并自动填充它们。
class Translation(BaseModel):
    """
    当用户需要翻译文本时，此工具用于识别并提取翻译任务所需的所有关键信息。
    """
    source_language: str = Field(description="待翻译文本的原始语种。例如：英文、中文。")
    target_language: str = Field(description="需要翻译成的目标语种。例如：中文、英文。")
    text_to_translate: str = Field(description="用户明确要求翻译的具体文本内容。")


# --- 2. 构建你的“智能体” ---
# 这个函数扮演智能体的角色。它接收用户输入，并利用LLM和工具来解析信息。
def translation_agent(user_query: str):
    """
    接收用户查询，并自动识别出翻译任务的各项参数。
    """
    print(f"--- 用户原始查询 ---\n'{user_query}'\n")

    # 这是智谱AI定义的工具格式
    tools = [
        {
            "type": "function",
            "function": {
                "name": "Translation",
                "description": "当用户需要翻译文本时，从用户查询中提取出源语言、目标语言和待翻译的文本。",
                "parameters": Translation.model_json_schema()
            }
        }
    ]

    # --- 3. 调用大模型进行“函数调用” ---
    # 我们将用户查询和工具定义一起发给LLM
    response = client.chat.completions.create(
        model="doubao-seed-2-0-lite-260215",
        messages=[{"role": "user", "content": user_query}],
        tools=tools,
        tool_choice="auto",  # 让模型自动决定是否使用工具
    )

    message = response.choices[0].message

    # --- 4. 解析结果 ---
    # 检查LLM是否决定使用我们定义的工具
    if not message.tool_calls:
        print("--- 智能体决策 ---")
        print("模型决定不使用任何工具，可能直接回答或追问。")
        return

    print("--- 智能体决策 ---")
    print("模型决定使用 'Translation' 工具来解析信息。\n")

    for tool_call in message.tool_calls:
        if tool_call.function.name == "Translation":
            # LLM返回的参数是一个JSON字符串，我们需要解析它
            tool_args = json.loads(tool_call.function.arguments)

            # 验证并打印提取出的信息
            print("--- 提取结果 ---")
            print(f"原始语种: {tool_args.get('source_language', '未能识别')}")
            print(f"目标语种: {tool_args.get('target_language', '未能识别')}")
            print(f"待翻译文本: {tool_args.get('text_to_translate', '未能识别')}")

            # 在真实应用中，你可以在这里调用一个真实的翻译API
            print("\n--- (模拟)执行翻译 ---")
            if tool_args.get('text_to_translate') == 'good！':
                print(f"翻译结果: '好！'")
            else:
                print("已完成翻译模拟。")


# --- 主程序入口 ---
if __name__ == "__main__":
    # 这是你的作业问题
    query = "帮我将good！翻译为中文"
    translation_agent(query)
