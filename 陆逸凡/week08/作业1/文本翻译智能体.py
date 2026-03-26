# 从pydantic导入BaseModel和Field，用于定义数据模型和数据验证
from pydantic import BaseModel, Field # 定义传入的数据请求格式
# 导入List类型，用于声明列表类型的字段
from typing import List
# 导入Literal，用于限制字段的取值只能是指定的几个字面量值
from typing_extensions import Literal

# 导入OpenAI库，用于调用大语言模型API
import openai
# 导入json库，用于处理JSON数据（虽然代码中未直接使用，但可能是预留）
import json

# 创建OpenAI客户端实例，配置API密钥和基础URL
client = openai.OpenAI(
    api_key="放你自己的api key", # 阿里云灵积模型服务的API密钥，获取地址：https://bailian.console.aliyun.com/?tab=model#/api-key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", # 阿里云灵积模型服务的API端点
)


"""
这个智能体（不是满足agent所有的功能），能自动生成tools的json，实现信息信息抽取
指定写的tool的格式
"""
# 定义信息抽取智能体类
class ExtractionAgent:
    # 初始化方法，接收模型名称作为参数
    def __init__(self, model_name: str):
        self.model_name = model_name  # 保存模型名称到实例变量

    # 调用方法，接收用户提示和响应模型，返回解析后的结构化数据
    def call(self, user_prompt, response_model):
        # 构造消息列表，将用户提示封装成OpenAI API所需格式
        messages = [
            {
                "role": "user",  # 消息角色为用户
                "content": user_prompt  # 消息内容为用户输入的提示文本
            }
        ]
        # 传入需要提取的内容，自己写了一个tool格式
        # 构造tools参数，定义函数调用格式，基于response_model的JSON schema
        tools = [
            {
                "type": "function",  # 工具类型为函数调用
                "function": {
                    "name": response_model.model_json_schema()['title'], # 工具名字，使用模型的title字段
                    "description": response_model.model_json_schema()['description'], # 工具描述，使用模型的description字段
                    "parameters": {  # 参数定义
                        "type": "object",  # 参数类型为对象
                        "properties": response_model.model_json_schema()['properties'], # 参数说明，使用模型的properties定义
                        "required": response_model.model_json_schema()['required'], # 必须要传的参数，使用模型的required字段
                    },
                }
            }
        ]
        #print("tools: ", tools)  # 被注释掉的调试语句，用于打印tools内容
        # 调用OpenAI聊天补全API
        response = client.chat.completions.create(
            model=self.model_name,  # 使用的模型名称
            messages=messages,  # 消息列表
            tools=tools,  # 工具定义列表
            tool_choice="auto",  # 让模型自动决定是否调用工具
        )
        try:
            # 提取的参数（json格式）
            # 从API响应中获取第一个选择的第一个工具调用的参数字符串
            arguments = response.choices[0].message.tool_calls[0].function.arguments

            # 参数转换为datamodel，关注想要的参数
            # 使用response_model的验证方法将JSON字符串解析为模型实例
            return response_model.model_validate_json(arguments)
        except:
            # 如果发生异常（如没有tool_calls或解析失败），打印错误信息和原始响应
            print('ERROR', response.choices[0].message)
            return None  # 返回None表示处理失败


# 定义文本数据模型，继承自BaseModel
class Text(BaseModel):
    """文本问答内容解析"""  # 模型的文档字符串，描述这个类的用途
    Original_language: List[str] = Field(description="原始语种")  # 原始语言列表，使用Field添加描述
    Target_languange: List[str] = Field(description="目标语种")  # 目标语言列表（注意这里有拼写错误：languange应为language）
    translate_text: str = Field(description="把需要翻译的文本放在这里")  # 待翻译的文本
    translated_text: str = Field(description="把翻译完成的文本放在这里")  # 翻译完成的文本

# 创建ExtractionAgent实例，使用"qwen-plus"模型，调用call方法进行翻译
result = ExtractionAgent(model_name = "qwen-plus").call('帮我将good！翻译为中文', Text)
# 打印结果
print(result)
# 预期输出示例：Original_language=['English'] Target_languange=['Chinese'] translate_text='good!' translated_text='好！'