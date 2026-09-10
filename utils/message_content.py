"""统一提取不同模型供应商返回的文本内容。"""
from __future__ import annotations

from typing import Any


def extract_text(content: Any) -> str:
    """将字符串、内容块字典或内容块列表归一化为可展示文本。"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        for key in ("text", "content", "output_text"):
            if key in content:
                return extract_text(content[key])
        return ""
    if isinstance(content, (list, tuple)):
        parts = [extract_text(item) for item in content]
        return "\n".join(part for part in parts if part)

    text = getattr(content, "text", None)
    if text is not None:
        return extract_text(text)
    nested_content = getattr(content, "content", None)
    if nested_content is not None and nested_content is not content:
        return extract_text(nested_content)
    return str(content)
