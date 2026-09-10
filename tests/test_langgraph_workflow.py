import asyncio
import unittest

from agent.intents import IntentClassifier
from agent.orchestrator import Orchestrator
from agent.react_agent import ReactAgent
from agent.skills import SkillRegistry
from agent.tools.contracts import EvidenceSource, ToolResult
from rag.vision_service import resolve_image


class FakeTool:
    def __init__(self, name, calls, result):
        self.name = name
        self.calls = calls
        self.result = result

    def invoke(self, *args, **kwargs):
        self.calls.append((self.name, args, kwargs))
        return self.result(*args, **kwargs)


class FakeTools:
    def __init__(self):
        self.calls = []
        self.sql = FakeTool("sql", self.calls, self._sql)
        self.rag = FakeTool("rag", self.calls, self._rag)
        self.vision = FakeTool("vision", self.calls, self._vision)

    @staticmethod
    def _sql(question):
        return ToolResult(
            tool_name="query_cases",
            ok=True,
            data={"sql": "SELECT 1", "columns": ["count"], "rows": [{"count": 1}]},
            sources=[EvidenceSource(source_type="database", source_id="results")],
        )

    @staticmethod
    def _rag(query, doc_type="", instructions=""):
        return ToolResult(
            tool_name="retrieve_knowledge",
            ok=True,
            data={"summary": "参考资料", "context": "案例与标准"},
            sources=[EvidenceSource(source_type="knowledge_base", source_id="kb-1")],
        )

    @staticmethod
    def _vision(image, question, context="", mode="assessment", instructions=""):
        return ToolResult(
            tool_name="analyze_image",
            ok=True,
            data={"image_id": image, "analysis": "图片事实", "mode": mode},
            sources=[EvidenceSource(source_type="image", source_id=image)],
        )


class RuntimeResolvingVisionTool:
    name = "vision"

    def __init__(self, calls):
        self.calls = calls
        self.resolved_images = []

    def invoke(self, image, question, context="", mode="assessment", instructions=""):
        self.calls.append((self.name, (image, question), {"mode": mode}))
        resolved = resolve_image(image)
        self.resolved_images.append((image, resolved))
        if not resolved:
            return ToolResult.failure("analyze_image", f"未找到图片 {image}", "VISION_ERROR")
        return ToolResult(
            tool_name="analyze_image",
            ok=True,
            data={"image_id": image, "analysis": "图片事实", "mode": mode},
            sources=[EvidenceSource(source_type="image", source_id=image)],
        )

class LangGraphWorkflowTests(unittest.TestCase):
    def make_orchestrator(self):
        tools = FakeTools()
        return Orchestrator(
            classifier=IntentClassifier(),
            skills=SkillRegistry.default(model=None),
            tools=tools,
            model=None,
        ), tools

    def test_compiled_graph_contains_business_skill_nodes(self):
        orchestrator, _ = self.make_orchestrator()
        nodes = set(orchestrator.graph.get_graph().nodes)
        self.assertIn("classify_intent", nodes)
        self.assertIn("create_plan", nodes)
        self.assertIn("route_plan", nodes)
        self.assertIn("skill_image_case_search", nodes)
        self.assertIn("skill_image_assessment", nodes)
        self.assertIn("compose_answer", nodes)

    def test_native_stream_exposes_graph_progress_and_final_state(self):
        orchestrator, _ = self.make_orchestrator()

        async def collect():
            return [
                event
                async for event in orchestrator.astream(
                    "擅自占用公路一共有多少条？",
                    thread_id="stream-test",
                )
            ]

        events = asyncio.run(collect())
        progress = [item["data"] for item in events if item["kind"] == "progress"]
        final = next(item["data"] for item in events if item["kind"] == "final")
        self.assertEqual(progress[0]["event"], "request_received")
        self.assertIn("reasoning_summary", [item["event"] for item in progress])
        self.assertIn("tool_observation", [item["event"] for item in progress])
        self.assertEqual(final.current_step, "completed")
        self.assertEqual(final.metadata["execution_policy"]["runtime"], "langgraph")

    def test_checkpointer_keeps_latest_state_by_thread(self):
        orchestrator, _ = self.make_orchestrator()
        thread_id = "checkpoint-test"
        state = orchestrator.handle(
            "擅自占用公路一共有多少条？",
            thread_id=thread_id,
        )
        snapshot = orchestrator.graph.get_state({"configurable": {"thread_id": thread_id}})
        self.assertEqual(snapshot.values["agent_state"].final_answer, state.final_answer)
        self.assertGreater(len(list(orchestrator.graph.get_state_history(
            {"configurable": {"thread_id": thread_id}}
        ))), 1)
        self.assertEqual(
            snapshot.values["agent_state"].recent_messages,
            [
                {"role": "user", "content": "擅自占用公路一共有多少条？"},
                {"role": "assistant", "content": state.final_answer},
            ],
        )

    def test_same_thread_starts_each_user_turn_with_fresh_execution_results(self):
        orchestrator, tools = self.make_orchestrator()
        thread_id = "two-turn-test"
        first = orchestrator.handle(
            "擅自占用公路一共有多少条？",
            thread_id=thread_id,
        )
        second = orchestrator.handle(
            "道路养护的判定标准是什么？",
            thread_id=thread_id,
        )
        self.assertEqual(first.selected_skills, ["case_query"])
        self.assertEqual(second.selected_skills, ["knowledge_query"])
        self.assertEqual([call[0] for call in tools.calls], ["sql", "rag"])

    def test_react_agent_wraps_native_langgraph_stream(self):
        orchestrator, _ = self.make_orchestrator()
        agent = ReactAgent(orchestrator=orchestrator, thread_id="agent-stream-test")

        async def collect():
            return [
                event
                async for event in agent.astream_events(
                    "擅自占用公路一共有多少条？",
                    version="v2",
                )
            ]

        events = asyncio.run(collect())
        custom = [item for item in events if item["event"] == "on_custom_event"]
        self.assertTrue(custom)
        self.assertEqual(custom[0]["metadata"]["runtime"], "langgraph")
        self.assertEqual(events[-1]["event"], "on_chain_end")
        self.assertEqual(events[-1]["data"]["status"], "completed")
        self.assertEqual(len(agent.history), 2)
        self.assertEqual(agent.history[0]["role"], "user")
        self.assertEqual(agent.history[1]["role"], "assistant")

    def test_dependent_multi_intent_routes_through_two_skill_nodes(self):
        orchestrator, tools = self.make_orchestrator()
        state = orchestrator.handle(
            "查找这张图片的历史案例，并根据案例判断有没有违法行为"
            "【上传图片：图片ID img_graph01】",
            thread_id="multi-intent-test",
        )
        self.assertEqual(state.selected_skills, ["image_case_search", "image_assessment"])
        self.assertEqual([call[0] for call in tools.calls], ["vision", "rag", "rag", "vision"])
        self.assertEqual(state.current_step, "completed")

    def test_each_uploaded_image_is_resolved_from_its_streamlit_runtime_context(self):
        orchestrator, tools = self.make_orchestrator()
        runtime_vision = RuntimeResolvingVisionTool(tools.calls)
        tools.vision = runtime_vision
        orchestrator.workflow.tools.vision = runtime_vision
        orchestrator.workflow.executor.tools.vision = runtime_vision

        first = orchestrator.handle(
            "查找这张图片的相似案例【上传图片：图片ID img_runtime01】",
            thread_id="runtime-images",
            uploaded_images={"img_runtime01": "data:image/png;base64,FIRST"},
        )
        second = orchestrator.handle(
            "判断这张图片是否违法【上传图片：图片ID img_runtime02】",
            thread_id="runtime-images",
            uploaded_images={"img_runtime02": "data:image/png;base64,SECOND"},
        )

        self.assertEqual(first.current_step, "completed")
        self.assertEqual(second.current_step, "completed")
        self.assertEqual(runtime_vision.resolved_images, [
            ("img_runtime01", "data:image/png;base64,FIRST"),
            ("img_runtime02", "data:image/png;base64,SECOND"),
        ])


if __name__ == "__main__":
    unittest.main()
