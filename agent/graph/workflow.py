"""Intent-Skill-Tool 的 LangGraph 主图。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from agent.display_labels import intent_label, skill_label
from agent.executor import PlanExecutor
from agent.intents import IntentClassifier, IntentType
from agent.memory import append_recent_turn, trim_recent_messages
from agent.planning.models import StepStatus
from agent.planning.planner import Planner
from agent.skills import SkillRegistry
from agent.skills.base import SkillResult
from agent.state import AgentState
from agent.tools.structured_tools import ToolRegistry
from rag.vision_service import use_uploaded_images
from utils.logger_handler import logger


ProgressCallback = Callable[[str, str], None] | None


class WorkflowState(TypedDict, total=False):
    """图内部状态；业务数据集中保存在可校验的 AgentState 中。"""

    agent_state: AgentState
    history: list[dict[str, str]]
    results: list[SkillResult]
    current_step_id: str | None


@dataclass
class WorkflowContext:
    """单次调用上下文，不写入 checkpoint。"""

    progress: ProgressCallback = None
    uploaded_images: dict[str, str] | None = None


class GraphProgress:
    """同时向旧回调和 LangGraph custom 流发布审计事件。"""

    def __init__(self, runtime: Runtime[WorkflowContext]):
        self.callback = runtime.context.progress if runtime.context else None
        self.writer = runtime.stream_writer

    def __call__(self, event: str, detail: str) -> None:
        if self.callback is not None:
            self.callback(event, detail)
        self.writer({"event": event, "detail": detail, "details": None})

    def emit_details(self, event: str, detail: str, details: dict[str, Any]) -> None:
        callback = self.callback
        emit_details = getattr(callback, "emit_details", None)
        if callable(emit_details):
            emit_details(event, detail, details)
        self.writer({"event": event, "detail": detail, "details": details})


class IntentSkillToolGraph:
    """把意图识别、任务规划和 Skill 执行编译为可持久化状态图。"""

    def __init__(
        self,
        classifier: IntentClassifier,
        skills: SkillRegistry,
        tools: ToolRegistry,
        model: Any | None,
        *,
        checkpointer: Any | None = None,
        general_answer: Callable[[str, list[dict[str, str]]], str],
        intent_details: Callable[[AgentState], dict[str, Any]],
        plan_details: Callable[[AgentState], dict[str, Any]],
        conclusion_details: Callable[[AgentState], dict[str, Any]],
    ):
        self.classifier = classifier
        self.skills = skills
        self.tools = tools
        self.model = model
        self.planner = Planner(skills)
        self.executor = PlanExecutor(skills, tools)
        self.general_answer = general_answer
        self.intent_details = intent_details
        self.plan_details = plan_details
        self.conclusion_details = conclusion_details
        self.checkpointer = checkpointer or InMemorySaver(
            serde=JsonPlusSerializer(
                allowed_msgpack_modules=[
                    ("agent.state", "AgentState"),
                    ("agent.intents.models", "IntentType"),
                    ("agent.planning.models", "StepStatus"),
                    ("agent.skills.base", "SkillResult"),
                ]
            )
        )
        self.graph = self._build().compile(
            checkpointer=self.checkpointer,
            name="intent_skill_tool_graph",
        )

    def _build(self) -> StateGraph:
        builder = StateGraph(WorkflowState, context_schema=WorkflowContext)
        builder.add_node("classify_intent", self._classify_intent)
        builder.add_node("request_clarification", self._request_clarification)
        builder.add_node("general_answer", self._general_answer)
        builder.add_node("create_plan", self._create_plan)
        builder.add_node("route_plan", self._route_plan)
        builder.add_node("compose_answer", self._compose_answer)
        builder.add_node("update_memory", self._update_memory)

        skill_routes: dict[str, str] = {}
        for intent in IntentType:
            if self.skills.get(intent) is None:
                continue
            node_name = f"skill_{intent.value}"
            builder.add_node(node_name, self._make_skill_node(intent))
            builder.add_edge(node_name, "route_plan")
            skill_routes[intent.value] = node_name

        builder.add_edge(START, "classify_intent")
        builder.add_conditional_edges(
            "classify_intent",
            self._route_after_intent,
            {
                "clarification": "request_clarification",
                "general_chat": "general_answer",
                "plan": "create_plan",
            },
        )
        builder.add_edge("request_clarification", "update_memory")
        builder.add_edge("general_answer", "update_memory")
        builder.add_edge("create_plan", "route_plan")
        builder.add_conditional_edges(
            "route_plan",
            self._route_to_skill,
            {**skill_routes, "compose": "compose_answer"},
        )
        builder.add_edge("compose_answer", "update_memory")
        builder.add_edge("update_memory", END)
        return builder

    @staticmethod
    def initial_state(query: str, history: list[dict[str, str]] | None) -> WorkflowState:
        recent_messages = trim_recent_messages(history)
        return {
            "agent_state": AgentState(
                user_query=query,
                recent_messages=recent_messages,
            ),
            "history": recent_messages,
            "results": [],
            "current_step_id": None,
        }

    def _classify_intent(
        self,
        workflow: WorkflowState,
        runtime: Runtime[WorkflowContext],
    ) -> WorkflowState:
        state = workflow["agent_state"]
        progress = GraphProgress(runtime)
        self._emit(state, progress, "request_received", "已接收请求")
        remembered_rounds = sum(
            1 for item in state.recent_messages if item.get("role") == "user"
        )
        self._emit(
            state,
            progress,
            "memory_loaded",
            f"已加载最近 {remembered_rounds} 轮对话上下文",
        )
        self._emit(state, progress, "intent_started", "正在识别用户意图和查询条件")
        state.intent = self.classifier.classify(state.user_query)
        state.current_step = "intent_classified"
        self._emit(
            state,
            progress,
            "intent_classified",
            f"意图识别完成：{intent_label(state.intent.primary_intent)}，置信度 {state.intent.confidence:.0%}",
        )
        self._emit_details(
            state,
            progress,
            "reasoning_summary",
            "意图识别依据",
            self.intent_details(state),
        )
        logger.info(
            "[langgraph] node=classify_intent intent=%s confidence=%.2f reason=%s",
            state.intent.primary_intent.value,
            state.intent.confidence,
            state.intent.reason,
        )
        return {**workflow, "agent_state": state}

    @staticmethod
    def _route_after_intent(workflow: WorkflowState) -> str:
        decision = workflow["agent_state"].intent
        assert decision is not None
        if decision.needs_clarification:
            return "clarification"
        if decision.primary_intent == IntentType.GENERAL_CHAT:
            return "general_chat"
        return "plan"

    def _request_clarification(
        self,
        workflow: WorkflowState,
        runtime: Runtime[WorkflowContext],
    ) -> WorkflowState:
        state = workflow["agent_state"]
        assert state.intent is not None
        state.final_answer = state.intent.clarification_question
        state.current_step = "completed"
        self._emit(state, GraphProgress(runtime), "clarification_requested", "需要补充信息")
        return {**workflow, "agent_state": state}

    def _general_answer(
        self,
        workflow: WorkflowState,
        runtime: Runtime[WorkflowContext],
    ) -> WorkflowState:
        state = workflow["agent_state"]
        state.final_answer = self.general_answer(state.user_query, workflow.get("history", []))
        state.current_step = "completed"
        self._emit(state, GraphProgress(runtime), "general_answer_generated", "一般问答生成完成")
        return {**workflow, "agent_state": state}

    def _create_plan(
        self,
        workflow: WorkflowState,
        runtime: Runtime[WorkflowContext],
    ) -> WorkflowState:
        state = workflow["agent_state"]
        assert state.intent is not None
        progress = GraphProgress(runtime)
        self._emit(state, progress, "planning_started", "正在拆解任务并生成执行计划")
        state.plan = self.planner.create_plan(state.intent)
        state.current_step = "planned"
        state.metadata["execution_policy"] = {
            "mode": "plan_execute",
            "runtime": "langgraph",
            "skill_react_max_iterations": state.plan.max_skill_iterations,
        }
        if not state.plan.steps:
            state.final_answer = "暂时无法识别合适的业务能力，请补充查询目标和范围。"
            state.current_step = "failed"
            state.errors.append("没有匹配的 Skill")
            return {**workflow, "agent_state": state}

        plan_labels = " → ".join(skill_label(step.skill_name) for step in state.plan.steps)
        self._emit(
            state,
            progress,
            "plan_created",
            f"执行计划已生成（{len(state.plan.steps)} 步）：{plan_labels}",
        )
        self._emit_details(
            state,
            progress,
            "reasoning_summary",
            "任务规划依据",
            self.plan_details(state),
        )
        return {**workflow, "agent_state": state}

    def _route_plan(
        self,
        workflow: WorkflowState,
        runtime: Runtime[WorkflowContext],
    ) -> WorkflowState:
        state = workflow["agent_state"]
        plan = state.plan
        if plan is None or not plan.steps:
            return {**workflow, "current_step_id": None}

        progress = GraphProgress(runtime)
        by_id = {step.id: step for step in plan.steps}
        for index, step in enumerate(plan.steps, 1):
            if step.status != StepStatus.PENDING:
                continue
            dependency_states = [by_id[item].status for item in step.depends_on]
            if any(status in {StepStatus.FAILED, StepStatus.SKIPPED} for status in dependency_states):
                step.status = StepStatus.SKIPPED
                step.error = "前置步骤失败"
                self._emit(
                    state,
                    progress,
                    "plan_step_skipped",
                    f"跳过第 {index} 步：{skill_label(step.skill_name)}（前置步骤未完成）",
                )
                continue
            if all(status == StepStatus.COMPLETED for status in dependency_states):
                return {**workflow, "agent_state": state, "current_step_id": step.id}

        return {**workflow, "agent_state": state, "current_step_id": None}

    @staticmethod
    def _route_to_skill(workflow: WorkflowState) -> str:
        step_id = workflow.get("current_step_id")
        state = workflow["agent_state"]
        if not step_id or state.plan is None:
            return "compose"
        step = next(item for item in state.plan.steps if item.id == step_id)
        return step.intent.value

    def _make_skill_node(self, intent: IntentType):
        def execute_skill(
            workflow: WorkflowState,
            runtime: Runtime[WorkflowContext],
        ) -> WorkflowState:
            state = workflow["agent_state"]
            plan = state.plan
            step_id = workflow.get("current_step_id")
            if plan is None or step_id is None:
                return workflow
            step = next(item for item in plan.steps if item.id == step_id)
            if step.intent != intent:
                raise ValueError(f"LangGraph 路由错误：{step.intent.value} -> {intent.value}")
            with use_uploaded_images(
                runtime.context.uploaded_images if runtime.context else None
            ):
                result = self.executor.execute_step(
                    state,
                    plan,
                    step_id,
                    progress=GraphProgress(runtime),
                )
            results = list(workflow.get("results", []))
            if result is not None:
                results.append(result)
            return {
                **workflow,
                "agent_state": state,
                "results": results,
                "current_step_id": None,
            }

        execute_skill.__name__ = f"execute_{intent.value}"
        return execute_skill

    def _compose_answer(
        self,
        workflow: WorkflowState,
        runtime: Runtime[WorkflowContext],
    ) -> WorkflowState:
        state = workflow["agent_state"]
        results = workflow.get("results", [])
        if state.final_answer is None:
            state.final_answer = self.executor.compose_answer(results)
        expected = len(state.plan.steps) if state.plan else 0
        all_ok = expected > 0 and len(results) == expected and all(result.ok for result in results)
        state.current_step = "completed" if all_ok else "failed"
        completion_detail = (
            ("处理完成" if all_ok else "处理未完成")
            if expected <= 1
            else ("全部任务处理完成" if all_ok else "部分任务未完成")
        )
        progress = GraphProgress(runtime)
        self._emit_details(
            state,
            progress,
            "analysis_summary",
            "结论形成依据",
            self.conclusion_details(state),
        )
        self._emit(
            state,
            progress,
            "orchestration_completed" if all_ok else "orchestration_incomplete",
            completion_detail,
        )
        return {**workflow, "agent_state": state}

    def _update_memory(
        self,
        workflow: WorkflowState,
        runtime: Runtime[WorkflowContext],
    ) -> WorkflowState:
        del runtime
        state = workflow["agent_state"]
        state.recent_messages = append_recent_turn(
            state.recent_messages,
            state.user_query,
            state.final_answer or "系统未生成有效回答。",
        )
        return {
            **workflow,
            "agent_state": state,
            "history": list(state.recent_messages),
        }

    @staticmethod
    def _emit(state: AgentState, callback: GraphProgress, event: str, detail: str) -> None:
        state.add_event(event, detail)
        callback(event, detail)

    @staticmethod
    def _emit_details(
        state: AgentState,
        callback: GraphProgress,
        event: str,
        detail: str,
        details: dict[str, Any],
    ) -> None:
        state.add_event(event, detail)
        callback.emit_details(event, detail, details)
