"""确定性 Plan Executor：按依赖顺序执行 Skill，并汇总可审计结果。"""
from __future__ import annotations

from collections.abc import Callable

from agent.audit_events import build_tool_call_details
from agent.display_labels import skill_label
from agent.planning.models import ExecutionPlan, StepStatus
from agent.skills.base import SkillResult
from agent.skills.registry import SkillRegistry
from agent.state import AgentState
from agent.tools.structured_tools import ToolRegistry


ProgressCallback = Callable[[str, str], None] | None


class PlanExecutor:
    def __init__(self, skills: SkillRegistry, tools: ToolRegistry):
        self.skills = skills
        self.tools = tools

    def execute(self, state: AgentState, plan: ExecutionPlan,
                progress: ProgressCallback = None) -> list[SkillResult]:
        results: list[SkillResult] = []
        for step in plan.steps:
            result = self.execute_step(state, plan, step.id, progress=progress)
            if result is not None:
                results.append(result)

        return results

    def execute_step(
        self,
        state: AgentState,
        plan: ExecutionPlan,
        step_id: str,
        progress: ProgressCallback = None,
    ) -> SkillResult | None:
        """执行一个计划步骤，供 LangGraph 节点和旧顺序入口共同复用。"""
        step = next((item for item in plan.steps if item.id == step_id), None)
        if step is None:
            raise ValueError(f"计划中不存在步骤 {step_id}")
        index = plan.steps.index(step) + 1
        by_id = {item.id: item for item in plan.steps}

        if any(by_id[dependency].status in {StepStatus.FAILED, StepStatus.SKIPPED}
               for dependency in step.depends_on):
            step.status = StepStatus.SKIPPED
            step.error = "前置步骤失败"
            self._emit(state, progress, "plan_step_skipped",
                       f"跳过第 {index} 步：{skill_label(step.skill_name)}（前置步骤未完成）")
            return None
        if any(by_id[dependency].status != StepStatus.COMPLETED
               for dependency in step.depends_on):
            step.status = StepStatus.SKIPPED
            step.error = "前置步骤未执行"
            return None

        skill = self.skills.get(step.intent)
        if skill is None:
            step.status = StepStatus.FAILED
            step.error = "没有匹配的 Skill"
            return None

        state.selected_skills.append(skill.name)
        state.current_step = "skill_selected"
        selection_detail = (
            f"已选择业务能力：{skill_label(skill.name)}"
            if len(plan.steps) == 1
            else f"执行计划第 {index}/{len(plan.steps)} 步：{skill_label(skill.name)}"
        )
        self._emit(state, progress, "skill_selected", selection_detail)
        step.status = StepStatus.RUNNING
        state.current_step = "executing"
        result = skill.execute(state, self.tools, progress=progress)
        self._merge_result(state, result)
        self._emit_tool_observations(state, progress, result)
        step.answer = result.answer
        step.status = StepStatus.COMPLETED if result.ok else StepStatus.FAILED
        if result.ok:
            self._emit(state, progress, "plan_step_completed",
                       f"第 {index} 步完成：{skill_label(skill.name)}")
        else:
            step.error = result.answer
            self._emit(state, progress, "plan_step_failed",
                       f"第 {index} 步未完成：{skill_label(skill.name)}")
        return result

    @staticmethod
    def _merge_result(state: AgentState, result: SkillResult) -> None:
        state.tool_results.extend(result.tool_results)
        state.evidence.extend(result.evidence)
        state.warnings.extend(result.warnings)
        state.artifacts.update(result.artifacts)

    @staticmethod
    def _emit_tool_observations(
        state: AgentState,
        callback: ProgressCallback,
        result: SkillResult,
    ) -> None:
        emit_details = getattr(callback, "emit_details", None)
        if not callable(emit_details):
            return
        emitted_ids = state.metadata.setdefault("_emitted_tool_result_ids", [])
        for tool_result in result.tool_results:
            result_id = id(tool_result)
            if result_id in emitted_ids:
                continue
            emitted_ids.append(result_id)
            details = build_tool_call_details(tool_result)
            title = f"调用详情：{details['tool_label']}"
            state.add_event("tool_observation", title)
            emit_details("tool_observation", title, details)

    @staticmethod
    def compose_answer(results: list[SkillResult]) -> str:
        if not results:
            return "暂时无法执行该请求，请补充具体业务目标。"
        if len(results) == 1:
            return results[0].answer
        sections = [
            f"## {skill_label(result.skill_name)}\n\n{result.answer}"
            for result in results
        ]
        return "\n\n---\n\n".join(sections)

    @staticmethod
    def _emit(state: AgentState, callback: ProgressCallback,
              event: str, detail: str) -> None:
        state.add_event(event, detail)
        if callback is not None:
            callback(event, detail)
