import os
import sys
import asyncio
import uuid
from collections.abc import AsyncIterator
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from agent.intents import IntentClassifier
from agent.audit_events import build_audit_event
from agent.memory import append_recent_turn, trim_recent_messages
from agent.orchestrator import Orchestrator
from agent.skills import SkillRegistry
from agent.tools.structured_tools import ToolRegistry
from model.factory import chat_model

class ReactAgent:
    """兼容旧入口的门面；内部由 LangGraph 运行 Intent-Skill-Tool。"""

    def __init__(
        self,
        orchestrator: Orchestrator | None = None,
        *,
        thread_id: str | None = None,
        history: list[dict[str, str]] | None = None,
    ):
        self.history: list[dict[str, str]] = trim_recent_messages(history)
        self.thread_id = thread_id or uuid.uuid4().hex
        self.orchestrator = orchestrator or Orchestrator(
            classifier=IntentClassifier(fallback_model=chat_model),
            skills=SkillRegistry.default(model=chat_model),
            tools=ToolRegistry.default(),
            model=chat_model,
        )

    def execute_stream(
        self,
        query: str,
        progress_callback=None,
        uploaded_images: dict[str, str] | None = None,
    ):
        handle_kwargs = {"history": self.history, "progress": progress_callback}
        if hasattr(self.orchestrator, "workflow"):
            handle_kwargs["thread_id"] = self.thread_id
            handle_kwargs["uploaded_images"] = uploaded_images
        state = self.orchestrator.handle(query, **handle_kwargs)
        answer = state.final_answer or "系统未生成有效回答。"
        self.history = append_recent_turn(self.history, query, answer)
        yield answer.strip() + "\n"

    async def astream_events(
        self,
        query: str,
        *,
        version: str = "v2",
        uploaded_images: dict[str, str] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """把 LangGraph 原生流转换为页面现有的 v2 审计事件契约。"""
        if version != "v2":
            raise ValueError("当前仅支持 astream_events(version='v2')")

        run_id = uuid.uuid4().hex
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        loop = asyncio.get_running_loop()
        sequence = 0

        yield {
            "event": "on_chain_start",
            "name": "intent_skill_tool_orchestrator",
            "run_id": run_id,
            "tags": ["auditable", "plan-execute"],
            "metadata": {"event_schema": "v2"},
            "data": {"input": query},
        }

        if hasattr(self.orchestrator, "astream"):
            final_state = None
            async for graph_event in self.orchestrator.astream(
                query,
                self.history,
                thread_id=self.thread_id,
                uploaded_images=uploaded_images,
            ):
                if graph_event.get("kind") == "progress":
                    raw = graph_event.get("data") or {}
                    sequence += 1
                    audit_event = build_audit_event(
                        raw.get("event", "graph_progress"),
                        raw.get("detail", "LangGraph 节点状态已更新"),
                        sequence,
                        raw.get("details"),
                    )
                    yield {
                        "event": "on_custom_event",
                        "name": "agent_progress",
                        "run_id": run_id,
                        "tags": [audit_event["category"], "langgraph"],
                        "metadata": {"event_schema": "v2", "runtime": "langgraph"},
                        "data": audit_event,
                    }
                elif graph_event.get("kind") == "final":
                    final_state = graph_event.get("data")

            answer = (
                getattr(final_state, "final_answer", None) or "系统未生成有效回答。"
            ).strip()
            self.history = append_recent_turn(self.history, query, answer)
            yield {
                "event": "on_chain_end",
                "name": "intent_skill_tool_orchestrator",
                "run_id": run_id,
                "tags": ["auditable", "plan-execute", "langgraph"],
                "metadata": {"event_schema": "v2", "runtime": "langgraph"},
                "data": {
                    "output": answer,
                    "status": getattr(final_state, "current_step", "failed"),
                    "selected_skills": getattr(final_state, "selected_skills", []),
                    "warning_count": len(getattr(final_state, "warnings", [])),
                    "error_count": len(getattr(final_state, "errors", [])),
                },
            }
            return

        def publish_event(
            event_name: str,
            detail: str,
            details: dict[str, Any] | None = None,
        ) -> None:
            nonlocal sequence
            sequence += 1
            audit_event = build_audit_event(event_name, detail, sequence, details)
            payload = {
                "event": "on_custom_event",
                "name": "agent_progress",
                "run_id": run_id,
                "tags": [audit_event["category"]],
                "metadata": {"event_schema": "v2"},
                "data": audit_event,
            }
            loop.call_soon_threadsafe(queue.put_nowait, payload)

        class ProgressPublisher:
            def __call__(self, event_name: str, detail: str) -> None:
                publish_event(event_name, detail)

            def emit_details(
                self,
                event_name: str,
                detail: str,
                details: dict[str, Any],
            ) -> None:
                publish_event(event_name, detail, details)

        progress_callback = ProgressPublisher()

        task = asyncio.create_task(asyncio.to_thread(
            self.orchestrator.handle,
            query,
            self.history,
            progress_callback,
        ))

        while not task.done() or not queue.empty():
            try:
                yield await asyncio.wait_for(queue.get(), timeout=0.05)
            except TimeoutError:
                continue

        state = await task
        while not queue.empty():
            yield queue.get_nowait()

        answer = (state.final_answer or "系统未生成有效回答。").strip()
        self.history = append_recent_turn(self.history, query, answer)
        yield {
            "event": "on_chain_end",
            "name": "intent_skill_tool_orchestrator",
            "run_id": run_id,
            "tags": ["auditable", "plan-execute"],
            "metadata": {"event_schema": "v2"},
            "data": {
                "output": answer,
                "status": state.current_step,
                "selected_skills": state.selected_skills,
                "warning_count": len(state.warnings),
                "error_count": len(state.errors),
            },
        }


if __name__ == '__main__':
    agent = ReactAgent()

    for chunk in agent.execute_stream("请统计2026年5月密云执法队的案件情况并生成执法报告草稿"):
        print(chunk, end="", flush=True)
