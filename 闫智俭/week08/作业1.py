from pydantic import BaseModel, Field
from typing import Optional
import openai
import json

# 定义翻译请求的数据结构
class TranslationRequest(BaseModel):
    """从用户输入中提取翻译任务的关键信息"""
    source_text: str = Field(description="需要翻译的原文文本内容")
    source_language: Optional[str] = Field(
        default=None, 
        description="原文的语种，如未明确指定可尝试推断，无法推断则为None"
    )
    target_language: str = Field(description="目标翻译语种")
    translation_context: Optional[str] = Field(
        default=None,
        description="翻译的上下文或特殊要求，如未提及则为None"
    )



class TranslationAgent:
    def __init__(self, model_name: str = "qwen-plus"):
        self.model_name = model_name
        self.client = openai.OpenAI(
            api_key="sk-f0ab3fca58044adcb75b5a60974549b3",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
    
    def extract_translation_info(self, user_input: str) -> TranslationRequest:
        """从用户输入中提取翻译信息"""
        # 动态生成工具描述
        tools = [
            {
                "type": "function",
                "function": {
                    "name": TranslationRequest.model_json_schema()['title'],
                    "description": TranslationRequest.model_json_schema()['description'],
                    "parameters": {
                        "type": "object",
                        "properties": TranslationRequest.model_json_schema()['properties'],
                        "required": ["source_text", "target_language"],
                    },
                }
            }
        ]
        
        messages = [
            {
                "role": "system",
                "content": """你是一个专业的翻译信息提取助手。请从用户输入中准确提取以下信息：
                1. 需要翻译的原文文本
                2. 原文的语种（如果用户明确指定）
                3. 目标翻译语种
                4. 翻译的上下文或特殊要求（如果有）
                
                注意：如果用户没有明确指定原文语种，你可以根据文本内容尝试推断常见语种（如英语、中文、日语等），
                如果无法推断，则将该字段设为None。"""
            },
            {
                "role": "user", 
                "content": user_input
            }
        ]
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "TranslationRequest"}},
        )
        
        try:
            arguments = response.choices[0].message.tool_calls[0].function.arguments
            return TranslationRequest.model_validate_json(arguments)
        except Exception as e:
            print(f'提取翻译信息出错: {e}')
            print('原始响应:', response.choices[0].message)
            return None
    
    def translate(self, user_input: str) -> dict:
        """完整翻译流程：提取信息 + 执行翻译"""
        # 1. 提取翻译信息
        trans_info = self.extract_translation_info(user_input)
        
        if not trans_info:
            return {"error": "无法提取翻译信息"}
        
        # 2. 构建翻译提示
        translation_prompt = f"""
        请将以下文本从{trans_info.source_language or '自动检测语种'}翻译为{trans_info.target_language}：
        
        原文：{trans_info.source_text}
        
        {f"上下文/要求：{trans_info.translation_context}" if trans_info.translation_context else ""}
        
        请提供准确、自然的翻译结果。
        """
        
        # 3. 调用模型进行翻译
        translation_response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "你是一个专业的翻译专家，提供准确、自然的翻译。"},
                {"role": "user", "content": translation_prompt}
            ],
            temperature=0.3
        )
        
        translated_text = translation_response.choices[0].message.content
        
        # 4. 返回完整结果
        return {
            "extracted_info": trans_info.dict(),
            "translation_result": translated_text,
            "translation_prompt": translation_prompt
        }


# 创建翻译智能体实例
translation_agent = TranslationAgent()

# 测试用例1：简单的翻译请求
test_cases = [
    "帮我将good！翻译为中文",
    "请把这句话翻译成法语：I love programming.",
    "将这段英文翻译成中文：The quick brown fox jumps over the lazy dog.",
    "Translate this Spanish text to English: ¡Hola! ¿Cómo estás?",
    "帮我把'你好，世界！'翻译成日文",
    "将这段中文翻译成英文，注意保持正式语气：我们很高兴邀请您参加下周的会议。",
    "翻译成德语：Good morning, how are you today?",
    "请将'谢谢'翻译成韩语",
    "把这个日语句子翻译成中文：今日は良い天気ですね。",
    "Translate to French with emphasis on formal tone: We appreciate your prompt response."
]

# 测试智能体
for i, test_case in enumerate(test_cases, 1):
    print(f"\n{'='*60}")
    print(f"测试用例 {i}: {test_case}")
    print('-'*60)
    
    result = translation_agent.translate(test_case)
    
    if "error" in result:
        print(f"错误: {result['error']}")
    else:
        info = result["extracted_info"]
        print(f"提取的翻译信息:")
        print(f"  - 原文: {info['source_text']}")
        print(f"  - 原语种: {info.get('source_language', '自动推断')}")
        print(f"  - 目标语种: {info['target_language']}")
        print(f"  - 上下文: {info.get('translation_context', '无')}")
        print(f"\n翻译结果: {result['translation_result']}")
