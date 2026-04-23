# 导入操作系统模块，用于设置环境变量
import os

# 设置阿里云百炼平台的API密钥（需要替换成自己的）
# https://bailian.console.aliyun.com/?tab=model#/api-key
os.environ["OPENAI_API_KEY"] = "自己的api key"
# 设置API的基础URL，指向阿里云百炼的兼容模式接口
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 导入异步IO模块，用于支持异步编程
import asyncio
# 导入UUID模块，用于生成唯一标识符
import uuid

# 从openai库导入响应内容完成事件和响应文本增量事件
from openai.types.responses import ResponseContentPartDoneEvent, ResponseTextDeltaEvent
# 从agents库导入Agent、流式事件、Runner、输入项类型、追踪功能
from agents import Agent, RawResponsesStreamEvent, Runner, TResponseInputItem, trace
# 导入设置默认OpenAI API类型和禁用追踪的函数
from agents import set_default_openai_api, set_tracing_disabled
# 设置默认的API类型为chat_completions（聊天补全模式）
set_default_openai_api("chat_completions")
# 禁用追踪功能（关闭调试追踪，提高运行效率）
set_tracing_disabled(True)

# 创建情感分类Agent（智能体）
Sentiment_Classification_agent = Agent(
    name="Sentiment_Classification_agent",  # Agent的名称
    model="qwen-max",  # 使用的模型：通义千问Max版本
    instructions="你是情感分类专家，擅长对文本进行情感分类，回答问题的时候先告诉我你是谁。",  # 系统指令：定义Agent的角色和行为
)

# 创建实体分类Agent（智能体）
Entity_Classification_agent = Agent(
    name="Entity_Classification_agent",  # Agent的名称
    model="qwen-max",  # 使用的模型：通义千问Max版本
    instructions="你是实体分类专家，擅长对文本进行实体识别，回答问题的时候先告诉我你是谁。",  # 系统指令：定义Agent的角色和行为
)

# 创建路由Agent（triage agent），负责根据用户请求分发给合适的专业Agent
triage_agent = Agent(
    name="triage_agent",  # Agent的名称
    model="qwen-max",  # 使用的模型：通义千问Max版本
    instructions="Handoff to the appropriate agent based on the language of the request.",  # 指令：根据请求语言转交给合适的Agent
    handoffs=[Sentiment_Classification_agent, Entity_Classification_agent],  # 可以转交的Agent列表（情感分类和实体分类）
)

# 定义主异步函数
async def main():
    # 为本次对话创建一个唯一ID（用于追踪会话），取UUID的前16位十六进制字符
    conversation_id = str(uuid.uuid4().hex[:16])

    # 获取用户的第一条输入
    msg = input("你好，我可以帮你对文本进行情感分类和实体识别，你还有什么问题？")
    # 设置当前使用的Agent为路由Agent
    agent = triage_agent
    # 初始化输入列表，将用户消息转换为标准格式
    inputs: list[TResponseInputItem] = [{"content": msg, "role": "user"}]

    # 无限循环，持续处理用户输入
    while True:
        # 使用追踪上下文管理器，记录这次路由操作
        with trace("Routing example", group_id=conversation_id):
            # 创建流式运行器，以流式方式执行Agent
            result = Runner.run_streamed(
                agent,  # 要运行的Agent
                input=inputs,  # 输入内容
            )
            # 异步遍历流式事件
            async for event in result.stream_events():
                # 如果不是原始响应流事件，则跳过
                if not isinstance(event, RawResponsesStreamEvent):
                    continue
                # 获取事件中的数据
                data = event.data
                # 如果是文本增量事件
                if isinstance(data, ResponseTextDeltaEvent):
                    # 打印增量的文本内容（不换行，实时显示）
                    print(data.delta, end="", flush=True)
                # 如果是响应内容完成事件
                elif isinstance(data, ResponseContentPartDoneEvent):
                    # 打印换行，表示一段内容结束
                    print("\n")

        # 将当前结果转换为输入列表格式，用于下一轮对话
        inputs = result.to_input_list()
        # 打印空行，美化输出格式
        print("\n")

        # 获取用户的下一轮输入
        user_msg = input("我可以继续帮你对文本进行情感分类和实体识别，你还有什么问题？")
        # 将新消息添加到输入列表中
        inputs.append({"content": user_msg, "role": "user"})
        # 注意：这里注释掉了Agent的更新，意味着每次都使用同一个Agent（路由Agent）

# 程序入口：如果当前脚本被直接运行（不是被导入）
if __name__ == "__main__":
    # 运行主异步函数
    asyncio.run(main())