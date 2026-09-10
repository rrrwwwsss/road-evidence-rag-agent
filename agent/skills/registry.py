"""Skill 注册与按 Intent 选择。"""
from __future__ import annotations

from agent.intents.models import IntentType
from agent.skills.base import Skill
from agent.skills.case_query import CaseQuerySkill
from agent.skills.image_assessment import ImageAssessmentSkill
from agent.skills.image_case_search import ImageCaseSearchSkill
from agent.skills.knowledge_query import KnowledgeQuerySkill
from agent.skills.report_generation import ReportGenerationSkill


class SkillRegistry:
    def __init__(self, skills: list[Skill]):
        self._by_intent = {
            intent: skill
            for skill in skills
            for intent in skill.supported_intents
        }

    def get(self, intent: IntentType) -> Skill | None:
        return self._by_intent.get(intent)

    @classmethod
    def default(cls, model=None) -> "SkillRegistry":
        return cls([
            CaseQuerySkill(),
            KnowledgeQuerySkill(),
            ImageCaseSearchSkill(),
            ImageAssessmentSkill(),
            ReportGenerationSkill(model=model),
        ])
