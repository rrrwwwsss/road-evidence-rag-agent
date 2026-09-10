import asyncio
import unittest

from agent.react_agent import ReactAgent
from agent.state import AgentState
from agent.audit_events import build_tool_call_details
from agent.tools.contracts import EvidenceSource, ToolResult


class FakeOrchestrator:
    def __init__(self):
        self.last_state = None

    def handle(self, query, history=None, progress=None):
        state = AgentState(user_query=query)
        progress("request_received", "已接收请求")
        progress("planning_started", "正在生成执行计划")
        if hasattr(progress, "emit_details"):
            progress.emit_details(
                "tool_observation",
                "调用详情：案件数据库查询",
                {"tool_name": "query_cases", "tool_label": "案件数据库查询", "ok": True},
            )
        progress("orchestration_completed", "处理完成")
        state.current_step = "completed"
        state.final_answer = "测试回答"
        state.selected_skills = ["case_query"]
        self.last_state = state
        return state


class EventStreamTests(unittest.TestCase):
    def test_astream_events_uses_v2_shape_and_auditable_details(self):
        async def collect():
            agent = ReactAgent(orchestrator=FakeOrchestrator())
            return [event async for event in agent.astream_events("测试请求", version="v2")]

        events = asyncio.run(collect())
        self.assertEqual(events[0]["event"], "on_chain_start")
        custom_events = [event for event in events if event["event"] == "on_custom_event"]
        self.assertEqual(len(custom_events), 4)
        self.assertEqual(custom_events[0]["name"], "agent_progress")
        self.assertEqual(custom_events[0]["data"]["category"], "系统处理")
        self.assertTrue(custom_events[1]["data"]["explanation"])
        self.assertNotIn("prompt", custom_events[1]["data"])
        self.assertEqual(custom_events[2]["data"]["details"]["tool_name"], "query_cases")
        self.assertEqual(events[-1]["event"], "on_chain_end")
        self.assertEqual(events[-1]["data"]["output"], "测试回答")
        self.assertEqual(events[-1]["data"]["status"], "completed")

    def test_rejects_unsupported_event_schema(self):
        async def collect():
            agent = ReactAgent(orchestrator=FakeOrchestrator())
            return [event async for event in agent.astream_events("测试请求", version="v1")]

        with self.assertRaises(ValueError):
            asyncio.run(collect())

    def test_sql_tool_details_include_command_and_limited_rows(self):
        result = ToolResult(
            tool_name="query_cases",
            ok=True,
            data={
                "sql": "SELECT * FROM results",
                "columns": ["id"],
                "rows": [{"id": index} for index in range(20)],
            },
            sources=[EvidenceSource(source_type="database", source_id="results")],
            latency_ms=12,
        )
        details = build_tool_call_details(result)
        self.assertEqual(details["command"], "SELECT * FROM results")
        self.assertEqual(details["result"]["row_count"], 20)
        self.assertEqual(len(details["result"]["rows_preview"]), 10)

    def test_rag_tool_details_exclude_full_context_and_redact_keys(self):
        result = ToolResult(
            tool_name="retrieve_knowledge",
            ok=True,
            data={
                "query": "Authorization: Bearer sk-1234567890",
                "doc_type": "case",
                "summary": "案例摘要",
                "context": "不应展示的完整知识库上下文",
            },
        )
        details = build_tool_call_details(result)
        self.assertNotIn("context", str(details))
        self.assertNotIn("sk-1234567890", str(details))


if __name__ == "__main__":
    unittest.main()
