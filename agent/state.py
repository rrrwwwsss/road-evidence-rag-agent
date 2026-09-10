"""Orchestrator 显式状态模型。"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from agent.intents.models import IntentDecision
from agent.planning.models import ExecutionPlan
from agent.tools.contracts import EvidenceSource, ToolResult


class StateEvent(BaseModel):
    event: str
    detail: str = ""
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


class AgentState(BaseModel):
    user_query: str
    recent_messages: list[dict[str, str]] = Field(default_factory=list)
    intent: IntentDecision | None = None
    plan: ExecutionPlan | None = None
    selected_skills: list[str] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)
    evidence: list[EvidenceSource] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    events: list[StateEvent] = Field(default_factory=list)
    current_step: Literal[
        "received", "intent_classified", "planned", "skill_selected", "executing", "completed", "failed"
    ] = "received"
    final_answer: str | None = None
    artifacts: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def add_event(self, event: str, detail: str = "") -> None:
        self.events.append(StateEvent(event=event, detail=detail))
