"""Intent-Skill-Tool 架构中 Tool 层的统一数据契约。"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class EvidenceSource(BaseModel):
    """一条可追溯证据。"""

    source_type: Literal["database", "knowledge_base", "image", "system"]
    source_id: str = ""
    title: str = ""
    excerpt: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """所有原子工具都使用的返回结构。"""

    tool_name: str
    ok: bool
    data: Any = None
    sources: list[EvidenceSource] = Field(default_factory=list)
    error_code: str | None = None
    error_message: str | None = None
    latency_ms: int = 0
    truncated: bool = False

    @classmethod
    def failure(cls, tool_name: str, error: Exception | str, code: str = "TOOL_ERROR") -> "ToolResult":
        return cls(
            tool_name=tool_name,
            ok=False,
            error_code=code,
            error_message=str(error),
        )
