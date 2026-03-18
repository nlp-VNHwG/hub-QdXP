from pydantic import BaseModel, Field
from typing import List
from typing_extensions import Literal
import openai

# 初始化OpenAI客户端（使用阿里云百炼兼容接口）
client = openai.OpenAI(
    api_key="sk-4806ae58c8de41848fd1153108c3d86c",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 定义通用提取智能体
class ExtractionAgent:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def call(self, user_prompt, response_model):
        messages = [{"role": "user", "content": user_prompt}]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": response_model.model_json_schema()['title'],
                    "description": response_model.model_json_schema()['description'],
                    "parameters": {
                        "type": "object",
                        "properties": response_model.model_json_schema()['properties'],
                        "required": response_model.model_json_schema()['required'],
                    },
                }
            }
        ]

        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        try:
            arguments = response.choices[0].message.tool_calls[0].function.arguments
            return response_model.model_validate_json(arguments)
        except:
            print('ERROR', response.choices[0].message)
            return None

# 定义翻译任务的输出结构
class Translation(BaseModel):
    """自动识别需要翻译的文本信息"""
    source_lang: str = Field(description="原始语种（例如：英语、中文、日语等）")
    target_lang: str = Field(description="目标语种（例如：英语、中文、日语等）")
    text: str = Field(description="待翻译的文本内容")

# 示例1：将英文翻译为中文
result = ExtractionAgent(model_name="qwen-plus").call(
    '帮我将good！翻译为中文', Translation
)
print("翻译结果1:", result)

# 示例2：将中文翻译为英文
result = ExtractionAgent(model_name="qwen-plus").call(
    '请把“你好世界”翻译成英语', Translation
)
print("翻译结果2:", result)

# 示例3：未明确指定语种，模型自动推断
result = ExtractionAgent(model_name="qwen-plus").call(
    '翻译bonjour', Translation
)
print("翻译结果3:", result)