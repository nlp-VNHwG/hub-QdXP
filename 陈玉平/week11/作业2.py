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
def multiply(
        num1: Annotated[float, "第一个数字"],
        num2: Annotated[float, "第二个数字"]
):
    """Multiply two numbers and return the result."""

    return num1 * num2


@mcp.tool
def generate_qr_code(
        content: Annotated[str, "The text or URL to encode in the QR code"],
        size: Annotated[int, "Size of the QR code image in pixels (default: 200)"] = 200,
        format: Annotated[str, "Image format: 'png' or 'svg' (default: 'png')"] = "png"
):
    """Generates a QR code image URL for the given content."""
    try:
        # 使用 goqr.me 免费 API
        encoded_content = requests.utils.quote(content)
        qr_url = f"https://api.qrserver.com/v1/create-qr-code/?size={size}x{size}&data={encoded_content}&format={format}"

        # 验证URL是否可访问
        test_response = requests.head(qr_url, timeout=5)
        if test_response.status_code == 200:
            return {
                "qr_code_url": qr_url,
                "content": content,
                "size": f"{size}x{size}",
                "format": format.upper()
            }
        else:
            return {"error": "QR码生成失败，请检查参数"}
    except Exception as e:
        return {"error": f"生成失败: {str(e)}"}

from datetime import datetime
@mcp.tool
def get_current_time(
        timezone: Annotated[str, "时区，如 'Asia/Shanghai', 'UTC'"] = "Asia/Shanghai"
):
    """Get the current time for a specified timezone."""

    # 简化示例，实际可用 pytz 处理时区
    now = datetime.now()

    return {
        "timezone": timezone,
        "current_time": now.strftime("%Y-%m-%d %H:%M:%S"),
        "timestamp": int(now.timestamp())
    }