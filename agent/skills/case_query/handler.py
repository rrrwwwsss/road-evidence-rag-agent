from __future__ import annotations

from pathlib import Path

from agent.intents.models import IntentType
from agent.skills.base import Skill, SkillResult
from agent.skills.common import ProgressCallback, format_sql_result, report_progress
from agent.skills.package import load_skill_package
from agent.state import AgentState
from agent.tools.structured_tools import ToolRegistry


PACKAGE = load_skill_package(Path(__file__).parent)


def _case_summary(result) -> str:
    if not result.ok:
        return f"查询失败：{result.error_message}"
    rows = result.data.get("rows", [])
    if not rows:
        return "根据当前查询条件，未查询到匹配案件。"

    if len(rows) == 1:
        row = rows[0]
        total = row.get("total_count", row.get("count", row.get("COUNT(*)")))
        confirmed = row.get("confirmed_violation_count")
        unconfirmed = row.get("unconfirmed_violation_count")
        if total is not None and confirmed is not None and unconfirmed is not None:
            return (
                f"根据当前查询条件，共查询到 **{total} 条**案件；其中，"
                f"**已确认违法 {confirmed} 条**，**未确认违法 {unconfirmed} 条**。"
            )
        if total is not None:
            return f"根据当前查询条件，共查询到 **{total} 条**案件记录。"
        facts = "，".join(f"{key}为 **{value}**" for key, value in row.items())
        return f"根据当前查询条件，{facts}。"

    return f"根据当前查询条件，查询返回 **{len(rows)} 条明细或分组结果**，具体数据见下表。"


def build_case_answer(result) -> str:
    summary = _case_summary(result)
    details = format_sql_result(result)
    note = (
        "`is_committed=1` 表示已确认违法；`is_committed=0` 表示未确认违法，"
        "不等同于已经确认合法或确认不违法。"
    )
    return f"## 查询结论\n\n{summary}\n\n## 数据明细\n\n{details}\n\n## 统计口径\n\n{note}"


class CaseQuerySkill(Skill):
    name = "case_query"
    supported_intents = {IntentType.CASE_QUERY}
    package = PACKAGE

    def execute(self, state: AgentState, tools: ToolRegistry,
                progress: ProgressCallback = None) -> SkillResult:
        report_progress(progress, "tool_started", "正在查询案件数据库")
        result = tools.sql.invoke(state.user_query)
        report_progress(progress, "tool_completed" if result.ok else "tool_failed",
                        "案件数据库查询完成" if result.ok else "案件数据库查询失败")
        report_progress(progress, "analysis_started", "正在分析并汇总查询结果")
        answer = build_case_answer(result)
        report_progress(progress, "analysis_completed", "查询结果分析完成")
        return SkillResult(
            skill_name=self.name,
            ok=result.ok,
            answer=answer,
            tool_results=[result],
            evidence=result.sources,
            metadata={"skill_package": self.package.name, "prompt_loaded": True},
        )
