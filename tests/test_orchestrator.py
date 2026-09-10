import unittest

from agent.intents import IntentClassifier
from agent.orchestrator import Orchestrator
from agent.skills import SkillRegistry
from agent.tools.contracts import EvidenceSource, ToolResult


class FakeTool:
    def __init__(self, name, calls, result_factory):
        self.name = name
        self.calls = calls
        self.result_factory = result_factory

    def invoke(self, *args, **kwargs):
        self.calls.append((self.name, args, kwargs))
        return self.result_factory(*args, **kwargs)


class FakeTools:
    def __init__(self):
        self.calls = []
        self.sql = FakeTool("sql", self.calls, self._sql_result)
        self.rag = FakeTool("rag", self.calls, self._rag_result)
        self.vision = FakeTool("vision", self.calls, self._vision_result)

    @staticmethod
    def _sql_result(question):
        return ToolResult(
            tool_name="query_cases",
            ok=True,
            data={
                "sql": "SELECT COUNT(*) AS count FROM results",
                "columns": ["count"],
                "rows": [{"count": 3}],
                "truncated": False,
            },
            sources=[EvidenceSource(source_type="database", source_id="results")],
        )

    @staticmethod
    def _rag_result(query, doc_type="", instructions=""):
        return ToolResult(
            tool_name="retrieve_knowledge",
            ok=True,
            data={
                "summary": "参考资料总结",
                "context": "判定标准与案例",
                "skill_prompt_applied": bool(instructions),
            },
            sources=[EvidenceSource(source_type="knowledge_base", source_id="standard.md")],
        )

    @staticmethod
    def _vision_result(image, question, context="", mode="assessment", instructions=""):
        return ToolResult(
            tool_name="analyze_image",
            ok=True,
            data={
                "image_id": image,
                "analysis": "图片客观特征描述",
                "mode": mode,
                "skill_prompt_applied": bool(instructions),
            },
            sources=[EvidenceSource(source_type="image", source_id=image)],
        )


class DetailedProgress:
    def __init__(self):
        self.events = []
        self.details = []

    def __call__(self, event, detail):
        self.events.append((event, detail))

    def emit_details(self, event, detail, details):
        self.details.append((event, detail, details))


class OrchestratorTests(unittest.TestCase):
    def make_orchestrator(self):
        tools = FakeTools()
        orchestrator = Orchestrator(
            classifier=IntentClassifier(),
            skills=SkillRegistry.default(model=None),
            tools=tools,
            model=None,
        )
        return orchestrator, tools

    def test_case_query_uses_only_sql(self):
        orchestrator, tools = self.make_orchestrator()
        progress_events = []
        state = orchestrator.handle(
            "擅自占用公路一共有多少条？",
            progress=lambda event, detail: progress_events.append((event, detail)),
        )
        self.assertEqual(state.selected_skills, ["case_query"])
        self.assertEqual([item[0] for item in tools.calls], ["sql"])
        self.assertEqual(state.current_step, "completed")
        self.assertIn("## 查询结论", state.final_answer)
        self.assertIn("共查询到 **3 条**", state.final_answer)
        self.assertIn("## 统计口径", state.final_answer)
        details = [detail for _, detail in progress_events]
        self.assertIn("正在分析并汇总查询结果", details)
        self.assertIn("查询结果分析完成", details)

    def test_detailed_progress_contains_reasoning_tool_and_conclusion_summaries(self):
        orchestrator, _ = self.make_orchestrator()
        progress = DetailedProgress()
        state = orchestrator.handle("擅自占用公路一共有多少条？", progress=progress)
        detail_events = [item[0] for item in progress.details]
        self.assertEqual(
            detail_events,
            ["reasoning_summary", "reasoning_summary", "tool_observation", "analysis_summary"],
        )
        self.assertEqual(progress.details[0][2]["reasoning_type"], "intent")
        self.assertEqual(progress.details[1][2]["reasoning_type"], "plan")
        self.assertEqual(progress.details[2][2]["tool_name"], "query_cases")
        self.assertEqual(progress.details[3][2]["reasoning_type"], "conclusion")
        self.assertEqual(progress.details[3][2]["execution_status"], "completed")
        self.assertEqual(state.current_step, "completed")

    def test_image_skill_forces_rag_before_vision(self):
        orchestrator, tools = self.make_orchestrator()
        progress_events = []
        state = orchestrator.handle(
            "图中是否属于道路养护？【上传图片：图片ID img_123abc】",
            progress=lambda event, detail: progress_events.append((event, detail)),
        )
        self.assertEqual(state.selected_skills, ["image_assessment"])
        self.assertEqual([item[0] for item in tools.calls], ["rag", "vision"])
        self.assertEqual(tools.calls[1][2]["context"], "判定标准与案例")
        details = [detail for _, detail in progress_events]
        self.assertIn("意图识别完成：图片违法与养护判定，置信度 96%", details)
        self.assertIn("已选择业务能力：图片违法与养护判定", details)
        self.assertFalse(any("image_assessment" in detail for detail in details))
        self.assertIn("正在检索判定标准和相似历史案例", details)
        self.assertIn("正在结合参考证据分析图片", details)
        self.assertEqual(details[-1], "处理完成")

    def test_image_case_search_describes_then_retrieves_without_judgement(self):
        orchestrator, tools = self.make_orchestrator()
        state = orchestrator.handle(
            "查找有没有类似的历史案例【上传图片：图片ID img_search01】"
        )
        self.assertEqual(state.selected_skills, ["image_case_search"])
        self.assertEqual([item[0] for item in tools.calls], ["vision", "rag"])
        self.assertEqual(tools.calls[0][2]["mode"], "retrieval")
        self.assertTrue(tools.calls[0][2]["instructions"])
        self.assertEqual(tools.calls[1][2]["doc_type"], "case")
        self.assertTrue(tools.calls[1][2]["instructions"])
        self.assertLessEqual(len(tools.calls[1][1][0]), 220)
        self.assertIn("不构成对当前图片违法或合法的认定", state.final_answer)

    def test_compound_image_request_executes_dependent_plan_and_reuses_cases(self):
        orchestrator, tools = self.make_orchestrator()
        state = orchestrator.handle(
            "查找这张图片的历史案例，并根据历史案例判断这个图片有没有擅自占用公路违法行为"
            "【上传图片：图片ID img_multi02】"
        )
        self.assertEqual(state.selected_skills, ["image_case_search", "image_assessment"])
        self.assertEqual([item[0] for item in tools.calls], ["vision", "rag", "rag", "vision"])
        self.assertEqual(tools.calls[1][2]["doc_type"], "case")
        self.assertEqual(tools.calls[2][2]["doc_type"], "standard")
        self.assertIn("判定标准与案例", tools.calls[3][2]["context"])
        self.assertTrue(state.plan.is_composite)
        self.assertEqual(state.plan.steps[1].depends_on, ["step_1"])
        self.assertEqual(state.metadata["execution_policy"]["mode"], "plan_execute")
        self.assertEqual(state.metadata["execution_policy"]["skill_react_max_iterations"], 3)
        self.assertIn("## 图片相似案例检索", state.final_answer)
        self.assertIn("## 图片违法与养护判定", state.final_answer)

    def test_compound_plan_stops_assessment_when_case_search_fails(self):
        orchestrator, tools = self.make_orchestrator()
        tools.rag.result_factory = lambda *args, **kwargs: ToolResult.failure(
            "retrieve_knowledge", "知识库不可用", "RAG_UNAVAILABLE"
        )
        state = orchestrator.handle(
            "查找这张图片的历史案例并据此判断这个图片有没有违法行为"
            "【上传图片：图片ID img_multi03】"
        )
        self.assertEqual([item[0] for item in tools.calls], ["vision", "rag"])
        self.assertEqual(state.plan.steps[1].status.value, "skipped")
        self.assertEqual(state.current_step, "failed")

    def test_image_without_upload_requests_real_upload(self):
        orchestrator, tools = self.make_orchestrator()
        state = orchestrator.handle("请判断这张图片是否属于道路养护")
        self.assertEqual(state.current_step, "failed")
        self.assertEqual(tools.calls, [])
        self.assertIn("上传", state.final_answer)

    def test_report_forces_sql_then_rag_without_pseudo_tool(self):
        orchestrator, tools = self.make_orchestrator()
        state = orchestrator.handle("统计2026年5月案件并生成执法报告")
        self.assertEqual(state.selected_skills, ["report_generation"])
        self.assertEqual([item[0] for item in tools.calls], ["sql", "rag"])
        self.assertNotIn("fill_context_for_report", [item[0] for item in tools.calls])
        self.assertIn("# 非现场取证报告草稿", state.final_answer)

    def test_report_stops_when_evidence_retrieval_fails(self):
        orchestrator, tools = self.make_orchestrator()
        tools.rag.result_factory = lambda *args, **kwargs: ToolResult.failure(
            "retrieve_knowledge", "知识库不可用", "RAG_UNAVAILABLE"
        )
        state = orchestrator.handle("统计2026年5月案件并生成执法报告")
        self.assertEqual([item[0] for item in tools.calls], ["sql", "rag"])
        self.assertEqual(state.current_step, "failed")
        self.assertIn("避免生成无来源事实", state.final_answer)


if __name__ == "__main__":
    unittest.main()
