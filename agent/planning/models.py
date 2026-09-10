"""Plan-Execute 的结构化计划模型。"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from agent.intents.models import IntentType


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class PlanStep(BaseModel):
    id: str
    intent: IntentType
    skill_name: str
    objective: str = ""
    depends_on: list[str] = Field(default_factory=list)
    status: StepStatus = StepStatus.PENDING
    answer: str = ""
    error: str | None = None


class ExecutionPlan(BaseModel):
    steps: list[PlanStep] = Field(default_factory=list)
    max_skill_iterations: int = Field(default=3, ge=1, le=5)

    @property
    def is_composite(self) -> bool:
        return len(self.steps) > 1
