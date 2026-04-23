"""作业2：三个自定义企业场景工具（纯本地逻辑）。"""

from __future__ import annotations

import re
import uuid
from datetime import date, timedelta
from typing import Annotated

from fastmcp import FastMCP

mcp = FastMCP(
    name="Homework-Custom-Tools",
    instructions="企业职能助手作业扩展：预算查询、会议室预约、周报大纲生成。",
)


@mcp.tool
def query_department_budget_remaining(
    department_code: Annotated[str, "部门编码，例如 HR、RD、SALES、OPS"],
) -> dict:
    """查询部门本年度剩余预算额度（模拟数据，用于演示工具调用）。"""
    mock = {
        "HR": {"total": 500_000, "used": 320_000},
        "RD": {"total": 2_000_000, "used": 1_450_000},
        "SALES": {"total": 800_000, "used": 610_000},
        "OPS": {"total": 600_000, "used": 200_000},
    }
    key = department_code.strip().upper()
    if key not in mock:
        return {
            "ok": False,
            "message": f"未知部门编码: {department_code}，可选: {', '.join(mock)}",
        }
    row = mock[key]
    remaining = row["total"] - row["used"]
    return {
        "ok": True,
        "department": key,
        "annual_budget": row["total"],
        "used_ytd": row["used"],
        "remaining": remaining,
        "unit": "CNY",
    }


@mcp.tool
def book_meeting_room(
    room_id: Annotated[str, "会议室编号，如 A-301、B-102"],
    date_str: Annotated[str, "日期 YYYY-MM-DD"],
    start_hour: Annotated[int, "开始小时 9-18（24 小时制整数）"],
    end_hour: Annotated[int, "结束小时，需大于 start_hour"],
) -> dict:
    """预约会议室（模拟）：校验格式并返回预订确认号。"""
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", date_str.strip()):
        return {"ok": False, "message": "日期格式须为 YYYY-MM-DD"}
    if not (9 <= start_hour < end_hour <= 18):
        return {
            "ok": False,
            "message": "时间须在 9-18 点之间，且 start_hour < end_hour",
        }
    booking_no = f"BK-{uuid.uuid4().hex[:8].upper()}"
    return {
        "ok": True,
        "booking_no": booking_no,
        "room_id": room_id.strip(),
        "date": date_str.strip(),
        "slot": f"{start_hour}:00 - {end_hour}:00",
        "status": "confirmed",
    }


@mcp.tool
def generate_weekly_report_outline(
    week_theme: Annotated[str, "本周工作主题或项目名称"],
    highlights: Annotated[str, "多条要点，可用逗号或换行分隔"],
) -> dict:
    """根据主题与要点生成「周报」Markdown 大纲结构（供用户再补充正文）。"""
    parts = re.split(r"[,，\n;；]+", highlights)
    bullets = [p.strip() for p in parts if p.strip()]
    if not bullets:
        bullets = ["（未提供要点，请补充后重新调用）"]

    monday = date.today() - timedelta(days=date.today().weekday())
    sunday = monday + timedelta(days=6)

    lines = [
        f"# 周报 · {week_theme.strip()}",
        f"## 周期 {monday.isoformat()} ~ {sunday.isoformat()}",
        "## 本周完成",
    ]
    for b in bullets:
        lines.append(f"- {b}")
    lines.extend(
        [
            "## 风险与阻塞",
            "- （待补充）",
            "## 下周计划",
            "- （待补充）",
        ]
    )
    outline = "\n".join(lines)
    return {"ok": True, "format": "markdown", "outline": outline}
