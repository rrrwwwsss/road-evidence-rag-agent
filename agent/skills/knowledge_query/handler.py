from __future__ import annotations

from pathlib import Path

from agent.intents.models import IntentType
from agent.skills.base import Skill, SkillResult
from agent.skills.common import ProgressCallback, report_progress
from agent.skills.package import load_skill_package
from agent.state import AgentState
from agent.tools.structured_tools import ToolRegistry


PACKAGE = load_skill_package(Path(__file__).parent)


class KnowledgeQuerySkill(Skill):
    name = "knowledge_query"
    supported_intents = {IntentType.KNOWLEDGE_QUERY}
    package = PACKAGE

    @staticmethod
    def _doc_type(question: str) -> str:
        if any(word in question for word in ("标准", "依据", "法规", "认定", "判断")):
            return "standard"
        if any(word in question for word in ("案例", "历史")):
            return "case"
        return ""

    def execute(self, state: AgentState, tools: ToolRegistry,
                progress: ProgressCallback = None) -> SkillResult:
        report_progress(progress, "tool_started", "正在检索判定标准和历史案例")
        result = tools.rag.invoke(
            state.user_query,
            doc_type=self._doc_type(state.user_query),
            instructions=self.package.prompt,
        )
        report_progress(progress, "tool_completed" if result.ok else "tool_failed",
                        f"知识库检索完成，获得 {len(result.sources)} 条参考资料" if result.ok else "知识库检索失败")
        answer = result.data["summary"] if result.ok else f"知识库检索失败：{result.error_message}"
        return SkillResult(
            skill_name=self.name,
            ok=result.ok,
            answer=answer,
            tool_results=[result],
            evidence=result.sources,
            metadata={"skill_package": self.package.name, "prompt_loaded": True},
            artifacts={
                "knowledge_query.rag_result": result,
                "knowledge_query.context": result.data.get("context", "") if result.ok else "",
            },
        )
