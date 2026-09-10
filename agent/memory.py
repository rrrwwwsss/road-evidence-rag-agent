"""会话级短期消息记忆。"""
from __future__ import annotations

from typing import Any


MAX_RECENT_ROUNDS = 8
MAX_RECENT_MESSAGES = MAX_RECENT_ROUNDS * 2
MAX_MESSAGE_CHARS = 6000


def _compact_content(value: Any) -> str:
    content = str(value or "").strip()
    if len(content) <= MAX_MESSAGE_CHARS:
        return content
    half = MAX_MESSAGE_CHARS // 2
    return content[:half] + "\n...[内容已截断]...\n" + content[-half:]


def trim_recent_messages(
    messages: list[dict[str, Any]] | None,
    *,
    max_rounds: int = MAX_RECENT_ROUNDS,
) -> list[dict[str, str]]:
    """只保留最近若干轮有效的用户/助手文本消息。"""
    normalized: list[dict[str, str]] = []
    for item in messages or []:
        role = item.get("role")
        content = _compact_content(item.get("content"))
        if role not in {"user", "assistant"} or not content:
            continue
        normalized.append({"role": role, "content": content})
    return normalized[-max(1, max_rounds) * 2:]


def append_recent_turn(
    messages: list[dict[str, Any]] | None,
    user_query: str,
    assistant_answer: str,
    *,
    max_rounds: int = MAX_RECENT_ROUNDS,
) -> list[dict[str, str]]:
    """追加本轮问答并再次应用记忆窗口。"""
    combined = list(messages or [])
    combined.extend([
        {"role": "user", "content": user_query},
        {"role": "assistant", "content": assistant_answer},
    ])
    return trim_recent_messages(combined, max_rounds=max_rounds)
