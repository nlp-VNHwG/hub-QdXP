import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
from agents import Agent, Runner, OpenAIProvider, function_tool

# 加载环境变量
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# 模型配置
model_provider = OpenAIProvider(api_key=api_key)
MODEL = "gpt-4o-mini"

# ====================== 【1】自定义Tool 1：企业成本计算 ======================
@function_tool
def calculate_business_cost(fixed_cost: float, variable_cost: float, quantity: int) -> str:
    """
    计算企业总成本：固定成本 + 单位变动成本 * 数量
    :param fixed_cost: 固定成本（租金、工资等）
    :param variable_cost: 单位变动成本
    :param quantity: 生产/销售数量
    :return: 总成本与明细
    """
    total = fixed_cost + variable_cost * quantity
    return (f"📊 企业成本计算结果\n"
            f"固定成本：{fixed_cost}\n"
            f"单位变动成本：{variable_cost} × 数量：{quantity}\n"
            f"✅ 总成本 = {total}")

# ====================== 【2】自定义Tool 2：查询当月工作日天数 ======================
@function_tool
def get_working_day_count(year: int, month: int) -> str:
    """
    计算指定年份、月份的工作日天数（周一到周五）
    :param year: 年份 如2025
    :param month: 月份 1-12
    :return: 工作日天数
    """
    try:
        current = datetime(year, month, 1)
        # 取下个月第一天
        if month == 12:
            next_month = datetime(year + 1, 1, 1)
        else:
            next_month = datetime(year, month + 1, 1)
        
        work_days = 0
        while current < next_month:
            if current.weekday() < 5:  # 0-4=周一到周五
                work_days += 1
            current += timedelta(days=1)
        
        return f"📅 {year}年{month}月 工作日天数：{work_days} 天"
    except:
        return "输入日期格式错误，请输入正确的年份和月份"

# ====================== 【3】自定义Tool 3：企业报告格式化 ======================
@function_tool
def format_company_report(report_title: str, content: str) -> str:
    """
    格式化企业报告，自动排版标题、分段
    :param report_title: 报告标题
    :param content: 报告正文
    :return: 格式化后的报告
    """
    return (
        f"=======================================\n"
        f"           【{report_title}】\n"
        f"=======================================\n"
        f"{content}\n"
        f"=======================================\n"
        f"报告已自动格式化完成"
    )

# ====================== 主Agent：企业职能助手 ======================
company_assistant = Agent(
    name="企业职能助手",
    description="企业办公智能助手，能自动调用成本计算、工作日查询、报告格式化工具",
    model=MODEL,
    provider=model_provider,
    instructions="""
    你是企业智能助手，根据用户的自然语言请求，自动选择对应的工具执行：
    1. 成本/费用/总价 → 调用 calculate_business_cost 成本计算工具
    2. 工作日/天数/考勤 → 调用 get_working_day_count 工作日查询工具
    3. 报告/排版/格式化/写报告 → 调用 format_company_report 报告格式化工具
    执行完成后，把工具返回结果整理后自然语言回复用户。
    """,
    tools=[calculate_business_cost, get_working_day_count, format_company_report]
)

# ====================== 交互对话框 ======================
if __name__ == "__main__":
    print("=" * 60)
    print("        企业职能助手（支持3个自定义Tool）")
    print("✅ 成本计算 | ✅ 工作日查询 | ✅ 报告格式化")
    print("输入 exit 退出")
    print("=" * 60)

    while True:
        user_input = input("\n💬 请输入你的需求：")
        if user_input.lower() == "exit":
            print("👋 退出程序")
            break
        
        # 运行主Agent，自动调用Tool
        result = Runner.run_sync(company_assistant, user_input)
        print("\n📌 助手回复：\n", result.final_output)
