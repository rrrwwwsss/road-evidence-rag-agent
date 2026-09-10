from __future__ import annotations

from pathlib import Path

from agent.intents.models import IntentType
from agent.policies import has_unconfirmed_case, requires_reference_evidence
from agent.skills.base import Skill, SkillResult
from agent.skills.common import ProgressCallback, report_progress
from agent.skills.package import load_skill_package
from agent.state import AgentState
from agent.tools.structured_tools import ToolRegistry


PACKAGE = load_skill_package(Path(__file__).parent)


class ImageAssessmentSkill(Skill):
    name = "image_assessment"
    supported_intents = {IntentType.IMAGE_ASSESSMENT}
    package = PACKAGE

    def execute(self, state: AgentState, tools: ToolRegistry,
                progress: ProgressCallback = None) -> SkillResult:
        assert state.intent is not None
        image_ids = state.intent.entities.image_ids
        if not image_ids:
            return SkillResult(skill_name=self.name, ok=False,
                               answer="请先通过聊天输入框上传需要分析的图片，然后重新提问。")

        prior_case_result = state.artifacts.get("image_case_search.rag_result")
        prior_case_context = state.artifacts.get("image_case_search.context", "")
        prior_standard_result = state.artifacts.get("knowledge_query.rag_result")
        prior_standard_context = state.artifacts.get("knowledge_query.context", "")
        is_composite = prior_case_result is not None or prior_standard_result is not None
        retrieval_detail = (
            "正在准备判定依据并复用前置步骤证据"
            if is_composite else "正在检索判定标准和相似历史案例"
        )
        report_progress(progress, "tool_started", retrieval_detail)
        if prior_standard_result is not None:
            rag_result = prior_standard_result
        else:
            rag_result = tools.rag.invoke(
                state.user_query,
                doc_type="standard" if prior_case_result is not None else "",
                instructions=self.package.prompt,
            )
        report_progress(progress, "tool_completed" if rag_result.ok else "tool_failed",
                        f"证据检索完成，获得 {len(rag_result.sources)} 条参考资料" if rag_result.ok else "证据检索失败")
        if not rag_result.ok and requires_reference_evidence(state.user_query):
            return SkillResult(
                skill_name=self.name,
                ok=False,
                answer=("暂时无法完成违法或养护行为判定：未取得必要的判定标准和参考案例。"
                        "请检查知识库服务后重试；涉及执法结论需人工复核。"),
                tool_results=[rag_result],
            )

        standard_context = prior_standard_context or (
            rag_result.data.get("context", "") if rag_result.ok else ""
        )
        context_parts = [part for part in (prior_case_context, standard_context) if part]
        context = "\n\n".join(context_parts)
        report_progress(progress, "tool_started", "正在结合参考证据分析图片")
        vision_result = tools.vision.invoke(
            image_ids[0], state.user_query, context=context,
            mode="assessment", instructions=self.package.prompt,
        )
        report_progress(progress, "tool_completed" if vision_result.ok else "tool_failed",
                        "图片分析完成" if vision_result.ok else "图片分析失败")
        warnings: list[str] = []
        answer = vision_result.data.get("analysis", "") if vision_result.ok else f"图片分析失败：{vision_result.error_message}"
        has_unconfirmed_reference = has_unconfirmed_case(rag_result)
        if prior_case_result is not None:
            has_unconfirmed_reference = has_unconfirmed_reference or has_unconfirmed_case(prior_case_result)
        if has_unconfirmed_reference:
            warning = (
                "检索到审核结果为“未确认违法”的相似案例；历史 AI 判断不能作为违法依据，"
                "当前图片不得仅凭模型输出标记为违法，需人工复核。"
            )
            warnings.append(warning)
            answer = f"{answer}\n\n> 风险提示：{warning}"

        results = [rag_result, vision_result]
        return SkillResult(
            skill_name=self.name,
            ok=vision_result.ok,
            answer=answer,
            tool_results=results,
            evidence=[source for result in results for source in result.sources],
            warnings=warnings,
            metadata={"skill_package": self.package.name, "prompt_loaded": True},
            artifacts={
                "image_assessment.standard_result": rag_result,
                "image_assessment.context": context,
            },
        )
