"""Skill 层基础契约。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

from pydantic import BaseModel, Field

from agent.intents.models import IntentType
from agent.state import AgentState
from agent.tools.contracts import EvidenceSource, ToolResult
from agent.tools.structured_tools import ToolRegistry


class SkillResult(BaseModel):
    skill_name: str
    ok: bool
    answer: str
    tool_results: list[ToolResult] = Field(default_factory=list)
    evidence: list[EvidenceSource] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, Any] = Field(default_factory=dict)


class Skill(ABC):
    name: str
    supported_intents: set[IntentType]

    @abstractmethod
    def execute(
        self,
        state: AgentState,
        tools: ToolRegistry,
        progress: Callable[[str, str], None] | None = None,
    ) -> SkillResult:
        raise NotImplementedError
