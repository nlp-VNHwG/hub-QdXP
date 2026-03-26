from pydantic import BaseModel, Field
from typing import Literal
import openai

# 初始化OpenAI客户端（对接阿里云通义千问）
client = openai.OpenAI(
    api_key="sk-f0ab3fca58044adcb75b5a60974549b3",  # 替换为自己的API Key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


# 翻译智能体核心类（复用参考代码的ExtractionAgent逻辑）
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
        # 基于Pydantic模型自动生成Tool描述
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

        # 调用大模型获取结构化结果
        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        try:
            # 提取并验证JSON格式的参数
            arguments = response.choices[0].message.tool_calls[0].function.arguments
            return response_model.model_validate_json(arguments)
        except Exception as e:
            print(f'ERROR: {e}, 响应内容: {response.choices[0].message}')
            return None


# 定义翻译信息抽取的Pydantic模型
class TranslationInfo(BaseModel):
    """文本翻译信息解析（提取原始语种、目标语种、待翻译文本）"""
    original_language: Literal["中文", "英文", "日语", "法语"] = Field(description="待翻译文本的原始语种")
    target_language: Literal["中文", "英文", "日语", "法语"] = Field(description="翻译的目标语种")
    text_to_translate: str = Field(description="需要翻译的文本内容")


# 扩展：可选 - 基于抽取的信息实现翻译（可选增强功能）
def translate_text(translation_info: TranslationInfo) -> str:
    """调用大模型完成实际翻译"""
    prompt = f"""请将【{translation_info.original_language}】文本「{translation_info.text_to_translate}」
    翻译成【{translation_info.target_language}】，仅返回翻译结果，不要额外解释。"""

    response = client.chat.completions.create(
        model="qwen-plus",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    return response.choices[0].message.content


# ---------------------- 测试用例 ----------------------
if __name__ == "__main__":
    # 测试场景：帮我讲good！翻译为中文
    user_prompt = "帮我讲good！翻译为中文"

    # 1. 抽取翻译核心信息（原始语种、目标语种、待翻译文本）
    agent = ExtractionAgent(model_name="qwen-plus")
    translation_info = agent.call(user_prompt, TranslationInfo)

    # 2. 打印抽取结果
    print("=== 翻译信息抽取结果 ===")
    print(f"原始语种: {translation_info.original_language}")
    print(f"目标语种: {translation_info.target_language}")
    print(f"待翻译文本: {translation_info.text_to_translate}")

    # 3. 执行翻译（可选增强）
    print("\n=== 翻译结果 ===")
    translated_text = translate_text(translation_info)
    print(translated_text)

# === 翻译信息抽取结果 ===
# 原始语种: 英文
# 目标语种: 中文
# 待翻译文本: good！
#
# === 翻译结果 ===
# 好！