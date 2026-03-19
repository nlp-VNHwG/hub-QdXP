from openai import Client
from pydantic import BaseModel, Field
from typing import List
from typing_extensions import Literal

import openai
import json

client = openai.OpenAI(
    api_key="sk-6ed8f79f823843a89a1f89122884afff",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
#这里并不需要继承BaseModel
class ExtractionAgent():
    def __init__(self,model_name:str):
        self.model_name = model_name

    def call(self,user_prompt, response_model):
        messages = [
            {
                'role' : 'user',
                'content' : user_prompt
            }
        ]

        #转化成JSON格式
        scheme = response_model.model_json_schema()

        tools = [
            {
                #这里是用来告诉大模型，给你的工具是一个函数
                "type": "function",
                #这个函数是用来干嘛的，有哪些特性
                "function":{
                    #这里会自动读取类名，就是调用的这个函数叫什么
                    "name" : scheme.get('title','default_name'),
                    #这个函数是干嘛用的
                    #这里其实就是你定义类那里所写的那段
                    "description" : scheme.get('description','default_description'),
                    #怎么用这个函数，需要哪些参数？
                    "parameters": {
                        "type" : "object",
                        "properties": scheme.get('properties',{}),
                        "required" : scheme.get('required',[]),
                    }
                }

            }
        ]

        try:
            response = client.chat.completions.create(
                model = self.model_name,
                messages = messages,
                #告诉大模型我给你这些工具，你自己去利用工具好好完成吧
                tools =tools,
                #自行选择tools
                tool_choice ='auto'
            )

            arguments = response.choices[0].message.tool_calls[0].function.arguments

            return response_model.model_validate_json(arguments)
        except Exception as e:
            print("提取失败，错误信息：",e)
            return None

#实例化
class TranslationAgent(BaseModel):
    """
    你需要完成文本翻译的任务
    """
    source_language: str = Field(description="等待翻译的原始语种，如果用户没有说明，你要自动去推导")
    target_language: str = Field(description="用户希望翻译的语种")
    text_to_translate: str = Field(description="需要被翻译的文本，去除’请帮我翻译一下‘之类的废话，只需要保留核心文本")
#执行最终的翻译任务
class finaltranslation(BaseModel):
    """
    你需要将输入的文本进行翻译，最终输出翻译后的句子
    """
    translated_text: str = Field(description="最终翻译好的目标语言文本内容 ")


if __name__ == '__main__':
    #调用Qwen-plus大模型
    agent = ExtractionAgent(model_name='qwen-plus')

    text = "我有一个美国来的客户，我想和他说‘欢迎你来到中国’，直接给出我英文就好了"

    print(f"用户输入: {text}")

    result = agent.call(text, TranslationAgent)

    if result:
        print(f"原始语种：{result.source_language}")
        print(f"目标语种：{result.target_language}")
        print(f"带翻译文本：{result.text_to_translate}")


    execute_prompt = (
        f"请将这段内容【{result.source_language}】翻译为【{result.target_language}】：\n"
        f"等待被翻译的文本【{result.text_to_translate}"
    )

    final_result = agent.call(execute_prompt, finaltranslation)

    print(f"翻译后的内容: {final_result.translated_text}")
