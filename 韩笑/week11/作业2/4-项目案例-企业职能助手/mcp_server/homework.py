import re
from typing import Annotated, Union
import requests
from typing import Literal
print("调用了homework工具")
TOKEN = "6d997a997fbf"

from fastmcp import FastMCP

mcp = FastMCP(
    name="Tools-MCP-Server",
    instructions="""This server contains some api of tools.""",
)

@mcp.tool
async def get_exercises_advice(args: dict)-> dict:
    """这是一个计算体脂率的工具，如果用户输入了年龄 身高 体重 性别 通过这个工具计算出体脂率 再给出合适的运动建议"""
    try:
        age = int(args.get("age"))
        weight = float(args.get("weight"))
        height = float(args.get("height"))

        sex = args.get("sex")
        print(f"用户输入年龄：{age},体重：{weight}，身高：{height}，性别：{sex}")
        bmi = weight / (height ** 2)
        fat = 0.0

        if age >= 18:
            if sex == '男' or sex == 'male':
                fat = 1.2 * bmi + 0.23 * age - 16.2
            else:
                fat = 1.2 * bmi + 0.23 * age - 16.2
        else:
            if sex == '男' or sex == 'male':
                fat = 1.51 * bmi - 0.70 * age - 2.2
            else:
                fat = 1.2 * bmi - 0.70 * age + 1.2
        # 假设根据体脂率生成建议
        advice = f"您的体脂率估算为: {fat:.2f}%。建议保持运动。"

        return {
            "bmi": bmi,
            "status": "Normal" if 18.5 <= bmi < 24 else "Need Attention",
            "advice": advice
        }
    except:
        return []

@mcp.tool
async def get_english_homework(args: dict)-> dict:
    """这是完成作业工具，如果需要完成英语作业调用这行工具"""
    try:
        homework = args.get("work_type")

        return {
            "advice": "已经完成英语作业"
        }
    except:
        return {
            "advice": "已经完成英语作业"
        }


@mcp.tool
async def get_math_homework(args: dict) -> dict:
    """这是完成作业工具，如果需要完成数学作业调用这行工具"""
    try:
        homework = args.get("work_type")

        return {
            "advice": "已经完成数学作业"
        }
    except:
        return {
            "advice": "已经完成数学作业"
        }