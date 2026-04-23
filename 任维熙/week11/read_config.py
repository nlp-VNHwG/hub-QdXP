"""从 homework 目录下的 config.json 读取应用配置。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AppConfig:
    api_key: str
    base_url: str = ""
    model: str = "qwen-flash"


def default_config_path() -> Path:
    return Path(__file__).resolve().parent / "config.json"


def default_config_payload() -> dict:
    return {
        "api_key": "",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-flash",
    }


def ensure_default_config(path: Path | None = None) -> bool:
    """
    若配置文件不存在，则创建带默认字段的 config.json（api_key 为空）。
    返回 True 表示本次新建了文件，False 表示文件原本就存在。
    """
    p = path or default_config_path()
    if p.is_file():
        return False
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(default_config_payload(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return True


def read_config(path: Path | None = None) -> AppConfig:
    p = path or default_config_path()
    created = ensure_default_config(p)

    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("config.json 根节点必须是 JSON 对象")

    api_key = str(raw.get("api_key", "")).strip()
    if not api_key:
        if created:
            hint = (
                f"已在 {p} 创建默认配置文件，请在该文件中填写 api_key 后重新运行。"
            )
        else:
            hint = f"请在 {p} 中填写 api_key 后重新运行。"
        print(hint, flush=True)
        raise ValueError(hint)

    base_url = str(raw.get("base_url", "") or "").strip()
    model = str(raw.get("model", "qwen-flash") or "qwen-flash").strip()

    return AppConfig(api_key=api_key, base_url=base_url, model=model)
