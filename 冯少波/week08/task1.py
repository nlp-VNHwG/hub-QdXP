from pydantic import BaseModel, Field
from typing import List
from typing_extensions import Literal
import openai

# 初始化 OpenAI 客户端
client = openai.OpenAI(
    api_key="sk-b2850aa9aff64528962998e0933d3912",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

class TranslationAgent:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def translate(self, user_prompt):
        class TranslationRequest(BaseModel):
            """文本翻译请求"""
            source_language: str = Field(description="原始语种，如'英文'、'中文'、'日文'等")
            target_language: str = Field(description="目标语种，如'中文'、'英文'、'法文'等")
            text_to_translate: str = Field(description="待翻译的文本内容")
        
        messages = [
            {
                "role": "user",
                "content": user_prompt
            }
        ]
        
        # 从Pydantic模型自动生成Tools JSON
        schema = TranslationRequest.model_json_schema()
        
        tools = [
            {
                "type": "function",
                "function": {
                    "name": schema['title'],
                    "description": schema['description'],
                    "parameters": {
                        "type": "object",
                        "properties": schema['properties'],
                        "required": schema['required'],
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
            result = TranslationRequest.model_validate_json(arguments)
            return result
        except Exception as e:
            return None

agent = TranslationAgent(model_name="qwen-plus")

result = agent.translate("帮我将good！翻译为中文")
if result:
    print(f"原始语种: {result.source_language}")
    print(f"目标语种: {result.target_language}")
    print(f"待翻译文本: {result.text_to_translate}")

