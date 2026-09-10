from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agent.intents.models import IntentType
from agent.skills.base import Skill, SkillResult
from agent.skills.common import ProgressCallback, format_sql_result, report_progress
from agent.skills.package import load_skill_package
from agent.state import AgentState
from agent.tools.contracts import ToolResult
from agent.tools.structured_tools import ToolRegistry
from utils.message_content import extract_text


PACKAGE = load_skill_package(Path(__file__).parent)


class ReportGenerationSkill(Skill):
    name = "report_generation"
    supported_intents = {IntentType.REPORT_GENERATION}
    package = PACKAGE

    def __init__(self, model: Any | None = None):
        self.model = model

    def execute(self, state: AgentState, tools: ToolRegistry,
                progress: ProgressCallback = None) -> SkillResult:
        report_progress(progress, "tool_started", "正在查询报告范围内的案件数据")
        sql_result = tools.sql.invoke(state.user_query)
        report_progress(progress, "tool_completed" if sql_result.ok else "tool_failed",
                        "案件统计完成" if sql_result.ok else "案件统计失败")
        if not sql_result.ok:
            return SkillResult(skill_name=self.name, ok=False,
                               answer=f"无法生成报告：案件数据库查询失败。{sql_result.error_message}",
                               tool_results=[sql_result])

        report_progress(progress, "tool_started", "正在检索报告所需的案件证据")
        rag_result = tools.rag.invoke(
            state.user_query,
            doc_type="case",
            instructions=self.package.prompt,
        )
        report_progress(progress, "tool_completed" if rag_result.ok else "tool_failed",
                        f"报告证据检索完成，获得 {len(rag_result.sources)} 条参考资料" if rag_result.ok else "报告证据检索失败")
        results = [sql_result, rag_result]
        if not rag_result.ok:
            return SkillResult(
                skill_name=self.name,
                ok=False,
                answer=("无法生成完整报告：已经取得数据库统计，但未取得可引用的案件证据。"
                        "为避免生成无来源事实，请恢复知识库后重试。"),
                tool_results=results,
                evidence=sql_result.sources,
            )

        report_progress(progress, "model_started", "正在根据事实和证据生成报告草稿")
        answer = self._compose(state.user_query, sql_result, rag_result)
        report_progress(progress, "model_completed", "报告草稿生成完成")
        return SkillResult(
            skill_name=self.name,
            ok=True,
            answer=answer,
            tool_results=results,
            evidence=[source for result in results for source in result.sources],
            metadata={"skill_package": self.package.name, "prompt_loaded": True},
        )

    def _compose(self, query: str, sql_result: ToolResult, rag_result: ToolResult) -> str:
        payload = {
            "用户要求": query,
            "数据库查询": sql_result.data,
            "知识库总结": rag_result.data.get("summary", ""),
            "证据来源": [source.model_dump() for source in sql_result.sources + rag_result.sources],
        }
        if self.model is not None:
            try:
                response = self.model.invoke([
                    SystemMessage(content=self.package.prompt),
                    HumanMessage(content=json.dumps(payload, ensure_ascii=False, default=str)),
                ])
                return extract_text(response.content)
            except Exception:
                pass

        return (
            f"# 非现场取证报告草稿\n\n## 摘要\n根据当前查询取得案件数据与知识库证据，"
            "以下内容需人工/执法部门复核。\n\n## 事实与证据\n"
            f"{format_sql_result(sql_result)}\n\n## 分析与判定理由\n"
            f"{rag_result.data.get('summary', '')}\n\n## 处置与取证建议\n"
            "1. 核对原始图片、拍摄时间和地点。\n2. 核对审批、施工或养护手续。\n"
            "3. 由执法人员结合现场证据作最终认定。\n\n## 参考资料\n"
            + "\n".join(f"- {source.title or source.source_id}" for source in rag_result.sources)
        )
