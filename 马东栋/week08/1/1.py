from pydantic import BaseModel, Field
from typing import List, Optional
import os
import openai
import json

client = openai.OpenAI(
    api_key=os.getenv("aliyunAPI_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

class TranslationInfo(BaseModel):
    """提取翻译相关信息"""
    source_text: str = Field(description="待翻译的文本")
    source_language: Optional[str] = Field(description="源语言，如中文、英文等")
    target_language: str = Field(description="目标语言，如中文、英文等")

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
        except Exception as e:
            print('ERROR:', e)
            print('Response:', response.choices[0].message)
            return None

class TranslationAgent:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.extraction_agent = ExtractionAgent(model_name)

    def translate(self, user_input):
        # 提取翻译信息
        extraction_result = self.extraction_agent.call(user_input, TranslationInfo)
        
        if not extraction_result:
            return "抱歉，无法提取翻译信息，请重新输入"
        
        # 构建翻译提示
        if extraction_result.source_language:
            prompt = f"请将以下{extraction_result.source_language}文本翻译成{extraction_result.target_language}：\n{extraction_result.source_text}"
        else:
            prompt = f"请将以下文本翻译成{extraction_result.target_language}：\n{extraction_result.source_text}"
        
        # 调用翻译
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages
        )
        
        return response.choices[0].message.content

# 测试
if __name__ == "__main__":
    agent = TranslationAgent(model_name="qwen-plus")
    
    # 测试案例1：明确指定源语言和目标语言
    result1 = agent.translate("请将'Hello, how are you?'从英文翻译成中文")
    print("测试1结果:", result1)
    
    # 测试案例2：只指定目标语言
    result2 = agent.translate("请将'你好，世界！'翻译成英文")
    print("测试2结果:", result2)
    
    # 测试案例3：复杂句子翻译
    result3 = agent.translate("请将'人工智能正在改变我们的生活方式'翻译成英文")
    print("测试3结果:", result3)