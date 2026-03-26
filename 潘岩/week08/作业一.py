from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from typing_extensions import Literal
from deep_translator import GoogleTranslator

import openai
import json

# 1. 配置OpenAI兼容客户端（通义千问）
client = openai.OpenAI(
    api_key="",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 2. 定义翻译指令解析的Pydantic模型（核心：识别原始语种、目标语种、待翻译文本）
class TranslationInfo(BaseModel):
    """翻译指令解析模型：自动识别翻译所需的核心信息"""
    intent: Literal["Translate"] = Field(description="用户意图，固定为Translate表示翻译")
    source_language: str = Field(description="原始语种（如：英语、中文）")
    target_language: str = Field(description="目标语种（如：中文、英语）")
    text_to_translate: str = Field(description="待翻译的文本内容")
    need_translate: bool = Field(default=True, description="是否需要执行翻译，默认True")

# 3. 封装智能体类（复用你提供的ExtractionAgent框架，增强翻译功能）
class TranslationAgent:
    def __init__(self, model_name: str = "qwen-plus"):
        self.model_name = model_name
        # 语种映射表（适配deep_translator的语种代码）
        self.lang_map = {
            "中文": "zh", "汉语": "zh",
            "英语": "en", "英文": "en",
            "日语": "ja", "韩语": "ko",
            "法语": "fr", "德语": "de"
        }

    def _parse_translation_intent(self, user_prompt: str) -> Optional[TranslationInfo]:
        """调用大模型解析用户指令，提取翻译核心信息"""
        messages = [
            {
                "role": "user",
                "content": f"""请解析以下翻译指令，提取关键信息并按照指定格式输出：
指令：{user_prompt}
要求：
1. intent固定为Translate；
2. 准确识别source_language（原始语种）、target_language（目标语种）、text_to_translate（待翻译文本）；
3. 示例：
   输入：帮我将good！翻译为中文 -> source_language=英语, target_language=中文, text_to_translate=good！
   输入：把"湖畔青石板上，一把油纸伞"翻译成英语 -> source_language=中文, target_language=英语, text_to_translate=湖畔青石板上，一把油纸伞
"""
            }
        ]

        # 生成Tools配置（基于Pydantic模型的JSON Schema）
        schema = TranslationInfo.model_json_schema()
        tools = [
            {
                "type": "function",
                "function": {
                    "name": schema["title"],
                    "description": schema["description"],
                    "parameters": {
                        "type": "object",
                        "properties": schema["properties"],
                        "required": schema.get("required", [])
                    }
                }
            }
        ]

        # 调用大模型
        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=tools,
                tool_choice="auto",  # 强制调用工具解析结构化数据
            )
            # 解析工具调用的参数
            arguments = response.choices[0].message.tool_calls[0].function.arguments
            return TranslationInfo.model_validate_json(arguments)
        except Exception as e:
            print(f"解析指令失败：{e}")
            print(f"原始响应：{response.choices[0].message}")
            return None

    def _translate(self, translation_info: TranslationInfo) -> str:
        """执行翻译（基于deep_translator）"""
        # 转换语种名称为代码
        src_code = self.lang_map.get(translation_info.source_language, "auto")
        tgt_code = self.lang_map.get(translation_info.target_language, "zh")
        
        try:
            # 调用Google翻译（无需API Key，适合测试）
            translator = GoogleTranslator(source=src_code, target=tgt_code)
            result = translator.translate(translation_info.text_to_translate)
            return result
        except Exception as e:
            return f"翻译执行失败：{str(e)}"

    def run(self, user_prompt: str) -> dict:
        """智能体主入口：解析指令 + 执行翻译"""
        # 步骤1：解析用户指令，提取翻译信息
        translation_info = self._parse_translation_intent(user_prompt)
        if not translation_info:
            return {"status": "failed", "msg": "指令解析失败"}
        
        # 步骤2：执行翻译
        translation_result = self._translate(translation_info)
        
        # 步骤3：返回结构化结果
        return {
            "status": "success",
            "parsed_info": {
                "原始语种": translation_info.source_language,
                "目标语种": translation_info.target_language,
                "待翻译文本": translation_info.text_to_translate
            },
            "翻译结果": translation_result
        }

if __name__ == "__main__":
    agent = TranslationAgent(model_name="qwen-plus")
    result1 = agent.run("帮我将good！翻译为中文")
    print("=== 测试用例1 ===")
    print(json.dumps(result1, ensure_ascii=False, indent=2))

    result2 = agent.run("把湖畔青石板上，一把油纸伞翻译成英语")
    print("\n=== 测试用例2 ===")
    print(json.dumps(result2, ensure_ascii=False, indent=2))
