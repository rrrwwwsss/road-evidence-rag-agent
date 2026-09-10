"""Intent -> Skill -> Tool 的 LangGraph 编排门面。"""
from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from typing import Any, Callable

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.display_labels import intent_label, skill_label
from agent.graph import IntentSkillToolGraph, WorkflowContext
from agent.intents import IntentClassifier, IntentType
from agent.skills import SkillRegistry
from agent.state import AgentState
from agent.tools.structured_tools import ToolRegistry
from utils.logger_handler import logger
from utils.prompt_loader import load_system_prompts
from utils.message_content import extract_text


ProgressCallback = Callable[[str, str], None] | None


class Orchestrator:
    def __init__(self, classifier: IntentClassifier, skills: SkillRegistry,
                 tools: ToolRegistry, model: Any | None = None):
        self.classifier = classifier
        self.skills = skills
        self.tools = tools
        self.model = model
        self.last_state: AgentState | None = None
        self.workflow = IntentSkillToolGraph(
            classifier=classifier,
            skills=skills,
            tools=tools,
            model=model,
            general_answer=self._general_answer,
            intent_details=self._intent_reasoning_details,
            plan_details=self._plan_reasoning_details,
            conclusion_details=self._conclusion_reasoning_details,
        )
        self.graph = self.workflow.graph

    def handle(self, query: str, history: list[dict[str, str]] | None = None,
               progress: ProgressCallback = None, *, thread_id: str | None = None,
               uploaded_images: dict[str, str] | None = None) -> AgentState:
        state = AgentState(user_query=query)
        try:
            output = self.graph.invoke(
                self.workflow.initial_state(query, history),
                config=self._config(thread_id),
                context=WorkflowContext(
                    progress=progress,
                    uploaded_images=uploaded_images,
                ),
            )
            state = output["agent_state"]
        except Exception as exc:
            logger.error("[orchestrator]执行失败：%s", exc, exc_info=True)
            state.errors.append(str(exc))
            state.current_step = "failed"
            state.final_answer = "系统处理请求时发生异常，请稍后重试。"
            self._emit(state, progress, "orchestration_failed", "系统处理发生异常")

        self.last_state = state
        return state

    async def astream(
        self,
        query: str,
        history: list[dict[str, str]] | None = None,
        *,
        thread_id: str | None = None,
        uploaded_images: dict[str, str] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """消费 LangGraph 原生 custom/values 流，并输出稳定的应用事件。"""
        state = AgentState(user_query=query)
        try:
            async for part in self.graph.astream(
                self.workflow.initial_state(query, history),
                config=self._config(thread_id),
                context=WorkflowContext(uploaded_images=uploaded_images),
                stream_mode=["custom", "values"],
                version="v2",
            ):
                part_type = part.get("type")
                if part_type == "custom":
                    yield {"kind": "progress", "data": part.get("data", {})}
                elif part_type == "values":
                    values = part.get("data", {})
                    candidate = values.get("agent_state") if isinstance(values, dict) else None
                    if isinstance(candidate, AgentState):
                        state = candidate
            self.last_state = state
            yield {"kind": "final", "data": state}
        except Exception as exc:
            logger.error("[orchestrator]流式执行失败：%s", exc, exc_info=True)
            state.errors.append(str(exc))
            state.current_step = "failed"
            state.final_answer = "系统处理请求时发生异常，请稍后重试。"
            self.last_state = state
            yield {
                "kind": "progress",
                "data": {"event": "orchestration_failed", "detail": "系统处理发生异常", "details": None},
            }
            yield {"kind": "final", "data": state}

    @staticmethod
    def _config(thread_id: str | None) -> dict[str, dict[str, str]]:
        return {"configurable": {"thread_id": thread_id or uuid.uuid4().hex}}

    def _general_answer(self, query: str, history: list[dict[str, str]]) -> str:
        if self.model is None:
            return "请说明需要查询案件、检索判定标准、分析图片，还是生成执法报告。"
        messages = [SystemMessage(content=load_system_prompts())]
        for item in history[-10:]:
            content = item.get("content", "")
            if content:
                message_type = HumanMessage if item.get("role") == "user" else AIMessage
                messages.append(message_type(content=content))
        messages.append(HumanMessage(content=query))
        response = self.model.invoke(messages)
        return extract_text(response.content)

    @staticmethod
    def _emit(state: AgentState, callback: ProgressCallback, event: str, detail: str) -> None:
        state.add_event(event, detail)
        if callback is not None:
            callback(event, detail)

    @staticmethod
    def _emit_details(
        state: AgentState,
        callback: ProgressCallback,
        event: str,
        detail: str,
        details: dict[str, Any],
    ) -> None:
        emit_details = getattr(callback, "emit_details", None)
        if not callable(emit_details):
            return
        state.add_event(event, detail)
        emit_details(event, detail, details)

    @staticmethod
    def _intent_reasoning_details(state: AgentState) -> dict[str, Any]:
        assert state.intent is not None
        entities = {
            key: value
            for key, value in state.intent.entities.model_dump().items()
            if value
        }
        return {
            "detail_type": "reasoning",
            "reasoning_type": "intent",
            "summary": state.intent.reason,
            "primary_intent": intent_label(state.intent.primary_intent),
            "secondary_intents": [intent_label(intent) for intent in state.intent.secondary_intents],
            "confidence": f"{state.intent.confidence:.0%}",
            "extracted_entities": entities,
            "goals": [
                {
                    "目标": goal.objective or intent_label(goal.intent),
                    "意图": intent_label(goal.intent),
                    "依赖": "、".join(intent_label(item) for item in goal.depends_on) or "无",
                }
                for goal in state.intent.goals
            ],
        }

    @staticmethod
    def _plan_reasoning_details(state: AgentState) -> dict[str, Any]:
        assert state.plan is not None
        step_labels = {step.id: skill_label(step.skill_name) for step in state.plan.steps}
        return {
            "detail_type": "reasoning",
            "reasoning_type": "plan",
            "summary": "根据目标依赖按确定顺序执行；前置步骤失败时停止依赖步骤。",
            "execution_mode": (
                "LangGraph Plan-Execute"
                if state.metadata.get("execution_policy", {}).get("runtime") == "langgraph"
                else "Plan-Execute"
            ),
            "max_skill_iterations": state.plan.max_skill_iterations,
            "steps": [
                {
                    "序号": index,
                    "业务能力": skill_label(step.skill_name),
                    "目标": step.objective,
                    "前置依赖": "、".join(step_labels.get(item, item) for item in step.depends_on) or "无",
                }
                for index, step in enumerate(state.plan.steps, 1)
            ],
        }

    @staticmethod
    def _conclusion_reasoning_details(state: AgentState) -> dict[str, Any]:
        seen_sources: set[tuple[str, str]] = set()
        sources: list[dict[str, str]] = []
        for source in state.evidence:
            key = (source.source_type, source.source_id)
            if key in seen_sources:
                continue
            seen_sources.add(key)
            sources.append({
                "来源类型": source.source_type,
                "来源": source.title or source.source_id or "未命名来源",
            })
            if len(sources) >= 10:
                break
        return {
            "detail_type": "reasoning",
            "reasoning_type": "conclusion",
            "summary": "最终回答由已完成 Skill 的结构化结果与可追溯证据汇总形成。",
            "execution_status": state.current_step,
            "selected_skills": [skill_label(name) for name in state.selected_skills],
            "tool_result_count": len(state.tool_results),
            "evidence_count": len(state.evidence),
            "sources": sources,
            "warnings": state.warnings,
            "errors": state.errors,
        }
