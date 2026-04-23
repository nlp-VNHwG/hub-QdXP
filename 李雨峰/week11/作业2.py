import random
import re
from typing import Annotated, Union
import requests
TOKEN = "6d997a997fbf"

from fastmcp import FastMCP
mcp = FastMCP(
    name="Tools-MCP-Server",
    instructions="""This server contains some api of tools.""",
)

@mcp.tool
def get_city_weather(city_name: Annotated[str, "The Pinyin of the city name (e.g., 'beijing' or 'shanghai')"]):
    """Retrieves the current weather data using the city's Pinyin name."""
    try:
        return requests.get(f"https://whyta.cn/api/tianqi?key={TOKEN}&city={city_name}").json()["data"]
    except:
        return []

@mcp.tool
def get_address_detail(address_text: Annotated[str, "City Name"]):
    """Parses a raw address string to extract detailed components (province, city, district, etc.)."""
    try:
        return requests.get(f"https://whyta.cn/api/tx/addressparse?key={TOKEN}&text={address_text}").json()["result"]
    except:
        return []

@mcp.tool
def get_tel_info(tel_no: Annotated[str, "Tel phone number"]):
    """Retrieves basic information (location, carrier) for a given telephone number."""
    try:
        return requests.get(f"https://whyta.cn/api/tx/mobilelocal?key={TOKEN}&phone={tel_no}").json()["result"]
    except:
        return []

@mcp.tool
def get_scenic_info(scenic_name: Annotated[str, "Scenic/tourist place name"]):
    """Searches for and retrieves information about a specific scenic spot or tourist attraction."""
    # https://apis.whyta.cn/docs/tx-scenic.html
    try:
        return requests.get(f"https://whyta.cn/api/tx/scenic?key={TOKEN}&word={scenic_name}").json()["result"]["list"]
    except:
        return []

@mcp.tool
def get_flower_info(flower_name: Annotated[str, "Flower name"]):
    """Retrieves the flower language (花语) and details for a given flower name."""
    # https://apis.whyta.cn/docs/tx-huayu.html
    try:
        return requests.get(f"https://whyta.cn/api/tx/huayu?key={TOKEN}&word={flower_name}").json()["result"]
    except:
        return []

@mcp.tool
def get_rate_transform(
    source_coin: Annotated[str, "The three-letter code (e.g., USD, CNY) for the source currency."], 
    aim_coin: Annotated[str, "The three-letter code (e.g., EUR, JPY) for the target currency."], 
    money: Annotated[Union[int, float], "The amount of money to convert."]
):
    """Calculates the currency exchange conversion amount between two specified coins."""
    try:
        return requests.get(f"https://whyta.cn/api/tx/fxrate?key={TOKEN}&fromcoin={source_coin}&tocoin={aim_coin}&money={money}").json()["result"]["money"]
    except:
        return []


@mcp.tool
def sentiment_classification(text: Annotated[str, "The text to analyze"]):
    """Classifies the sentiment of a given text."""
    positive_keywords_zh = ['喜欢', '赞', '棒', '优秀', '精彩', '完美', '开心', '满意']
    negative_keywords_zh = ['差', '烂', '坏', '糟糕', '失望', '垃圾', '厌恶', '敷衍']

    positive_pattern = '(' + '|'.join(positive_keywords_zh) + ')'
    negative_pattern = '(' + '|'.join(negative_keywords_zh) + ')'

    positive_matches = re.findall(positive_pattern, text)
    negative_matches = re.findall(negative_pattern, text)

    count_positive = len(positive_matches)
    count_negative = len(negative_matches)

    if count_positive > count_negative:
        return "积极 (Positive)"
    elif count_negative > count_positive:
        return "消极 (Negative)"
    else:
        return "中性 (Neutral)"


@mcp.tool
def query_salary_info(user_name: Annotated[str, "用户名"]):
    """Query user salary baed on the username."""

    # TODO 基于用户名，在数据库中查询，返回数据库查询结果

    if len(user_name) == 2:
        return 1000
    elif len(user_name) == 3:
        return 2000
    else:
        return 3000

@mcp.tool
def get_current_time(
    timezone: Annotated[str, "Timezone string, e.g., 'Asia/Shanghai', 'UTC'"] = "Asia/Shanghai"
):
    """Returns the current time in the specified timezone (default: Asia/Shanghai)."""
    try:
        # 尝试使用 pytz 如果可用，否则回退到本地时间
        try:
            import pytz
            tz = pytz.timezone(timezone)
            now = datetime.now(tz)
        except ImportError:
            # 如果没有 pytz，简单返回本地时间并注明
            now = datetime.now()
            return f"当前本地时间：{now.strftime('%Y-%m-%d %H:%M:%S')} (pytz未安装，显示系统时间)"
        return now.strftime("%Y-%m-%d %H:%M:%S")
    except Exception as e:
        return f"获取时间失败：{str(e)}"


@mcp.tool
def calculate(
    a: Annotated[Union[int, float], "第一个数字"],
    b: Annotated[Union[int, float], "第二个数字"],
    operation: Annotated[str, "运算类型，可选：add, subtract, multiply, divide"] = "add"
):
    """Performs basic arithmetic operations (add, subtract, multiply, divide) on two numbers."""
    try:
        if operation == "add":
            return a + b
        elif operation == "subtract":
            return a - b
        elif operation == "multiply":
            return a * b
        elif operation == "divide":
            if b == 0:
                return "错误：除数不能为零"
            return a / b
        else:
            return "错误：不支持的运算类型，请使用 add, subtract, multiply, divide"
    except Exception as e:
        return f"计算失败：{str(e)}"


@mcp.tool
def get_random_motivation():
    """Returns a random motivational quote (in Chinese) to inspire the user."""
    quotes = [
        "坚持就是胜利，每一步都算数。",
        "今天的努力，是明天的底气。",
        "不要等待机会，而要创造机会。",
        "相信自己的潜力，你比你想象的更强大。",
        "行动是治愈恐惧的良药。",
        "失败只是成功路上的减速带。",
        "每天进步一点点，终将抵达远方。"
    ]
    return random.choice(quotes)