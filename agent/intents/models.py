"""Intent 层结构化模型。"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class IntentType(str, Enum):
    CASE_QUERY = "case_query"
    KNOWLEDGE_QUERY = "knowledge_query"
    IMAGE_CASE_SEARCH = "image_case_search"
    IMAGE_ASSESSMENT = "image_assessment"
    REPORT_GENERATION = "report_generation"
    GENERAL_CHAT = "general_chat"
    UNKNOWN = "unknown"


class QueryEntities(BaseModel):
    dates: list[str] = Field(default_factory=list)
    time_range: str | None = None
    units: list[str] = Field(default_factory=list)
    locations: list[str] = Field(default_factory=list)
    violation_types: list[str] = Field(default_factory=list)
    case_ids: list[str] = Field(default_factory=list)
    image_ids: list[str] = Field(default_factory=list)


class IntentGoal(BaseModel):
    """用户在一次请求中希望完成的一个独立业务目标。"""

    intent: IntentType
    objective: str = ""
    depends_on: list[IntentType] = Field(default_factory=list)


class IntentDecision(BaseModel):
    primary_intent: IntentType
    secondary_intents: list[IntentType] = Field(default_factory=list)
    goals: list[IntentGoal] = Field(default_factory=list)
    entities: QueryEntities = Field(default_factory=QueryEntities)
    confidence: float = Field(ge=0, le=1)
    needs_clarification: bool = False
    clarification_question: str | None = None
    reason: str = ""
