from pydantic import BaseModel, Field # 定义传入的数据请求格式
from typing import List
from typing_extensions import Literal

import openai
import json

client = openai.OpenAI(
    api_key="sk-9f06aac1b31541958699954fe1ca8432",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

class ExtractionAgent:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def call(self, user_prompt, response_model):
        messages = [
            {
                "role": "user",
                "content": user_prompt
            }
        ]
        # 传入需要提取的内容，自己写了一个tool格式
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "Ticket",
                    "description": "根据用户提供的原始内容翻译成目标语言",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "origin_language": {
                                "description": "原始文本的语言类型",
                                "title": "language type",
                                "type": "string",
                            },
                            "aim_language": {
                                "description": "目标语言的语言类型",
                                "title": "language type",
                                "type": "string",
                            },
                            "text": {
                                "description": "文本中待翻译的内容",
                                "title": "text waiting to be translate",
                                "type": "string",
                            },
                        },
                        "required": ["origin_language", "aim_language", "text"],
                    },
                },
            },
            
        ]

        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        try:
            # 提取的参数（json格式）
            arguments = response.choices[0].message.tool_calls[0].function.arguments

            # 参数转换为datamodel，关注想要的参数
            return response_model.model_validate_json(arguments)
        except:
            print('ERROR', response.choices[0].message)
            return None


class Text(BaseModel):
    origin_language: str = Field(description="原始语言")
    aim_language: str = Field(description="目标语言")
    text: str = Field(description="需要翻译的文本")

result = ExtractionAgent(model_name = "qwen-plus").call('你能帮我把good翻译成中文吗', Text)
print(result)