"""把结构化目标转换为可校验、可执行的 Skill 依赖计划。"""
from __future__ import annotations

from agent.intents.models import IntentDecision, IntentGoal, IntentType
from agent.planning.models import ExecutionPlan, PlanStep
from agent.skills.registry import SkillRegistry


class Planner:
    def __init__(self, skills: SkillRegistry, max_skill_iterations: int = 3):
        self.skills = skills
        self.max_skill_iterations = max_skill_iterations

    def create_plan(self, decision: IntentDecision) -> ExecutionPlan:
        goals = decision.goals or [
            IntentGoal(intent=decision.primary_intent, objective=decision.reason)
        ]
        steps: list[PlanStep] = []
        step_id_by_intent: dict[IntentType, str] = {}

        for index, goal in enumerate(goals, 1):
            skill = self.skills.get(goal.intent)
            if skill is None:
                continue
            step_id = f"step_{index}"
            step_id_by_intent[goal.intent] = step_id
            steps.append(PlanStep(
                id=step_id,
                intent=goal.intent,
                skill_name=skill.name,
                objective=goal.objective,
            ))

        for step, goal in zip(steps, [goal for goal in goals if self.skills.get(goal.intent)]):
            step.depends_on = [
                step_id_by_intent[intent]
                for intent in goal.depends_on
                if intent in step_id_by_intent
            ]

        self._validate(steps)
        return ExecutionPlan(steps=steps, max_skill_iterations=self.max_skill_iterations)

    @staticmethod
    def _validate(steps: list[PlanStep]) -> None:
        ids = {step.id for step in steps}
        completed: set[str] = set()
        for step in steps:
            if any(dependency not in ids for dependency in step.depends_on):
                raise ValueError(f"计划步骤 {step.id} 引用了不存在的依赖")
            if any(dependency not in completed for dependency in step.depends_on):
                raise ValueError(f"计划步骤 {step.id} 的依赖顺序无效")
            completed.add(step.id)
