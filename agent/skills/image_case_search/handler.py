from __future__ import annotations

import re
from pathlib import Path

from agent.intents.models import IntentType
from agent.skills.base import Skill, SkillResult
from agent.skills.common import ProgressCallback, report_progress
from agent.skills.package import load_skill_package
from agent.state import AgentState
from agent.tools.structured_tools import ToolRegistry


PACKAGE = load_skill_package(Path(__file__).parent)


def build_case_retrieval_query(user_query: str, visual_description: str,
                               max_chars: int = 220) -> str:
    """优先使用视觉模型给出的检索关键词，避免把整篇分析发送给 Dify。"""
    keyword_match = re.search(
        r"检索关键词[：:]\s*(.+)",
        visual_description,
        flags=re.IGNORECASE,
    )
    visual_terms = keyword_match.group(1).strip() if keyword_match else visual_description
    visual_terms = " ".join(visual_terms.split())
    clean_question = re.sub(r"【上传图片：图片ID\s+img_[A-Za-z0-9]+】", "", user_query)
    clean_question = " ".join(clean_question.split()).strip()
    candidate = f"{clean_question}；图片特征：{visual_terms}" if clean_question else visual_terms
    return candidate[:max_chars]


class ImageCaseSearchSkill(Skill):
    name = "image_case_search"
    supported_intents = {IntentType.IMAGE_CASE_SEARCH}
    package = PACKAGE

    def execute(self, state: AgentState, tools: ToolRegistry,
                progress: ProgressCallback = None) -> SkillResult:
        assert state.intent is not None
        image_ids = state.intent.entities.image_ids
        if not image_ids:
            return SkillResult(skill_name=self.name, ok=False,
                               answer="请先通过聊天输入框上传需要比对的图片，然后重新提问。")

        report_progress(progress, "tool_started", "正在提取图片中的客观场景特征")
        vision_result = tools.vision.invoke(
            image_ids[0], state.user_query, context="", mode="retrieval",
            instructions=self.package.prompt,
        )
        report_progress(progress, "tool_completed" if vision_result.ok else "tool_failed",
                        "图片特征提取完成" if vision_result.ok else "图片特征提取失败")
        if not vision_result.ok:
            return SkillResult(skill_name=self.name, ok=False,
                               answer=f"无法查找相似案例：{vision_result.error_message}",
                               tool_results=[vision_result])

        visual_description = vision_result.data.get("analysis", "")
        retrieval_query = build_case_retrieval_query(state.user_query, visual_description)
        report_progress(progress, "tool_started", "正在根据图片特征检索相似历史案例")
        rag_result = tools.rag.invoke(
            retrieval_query,
            doc_type="case",
            instructions=self.package.prompt,
        )
        report_progress(progress, "tool_completed" if rag_result.ok else "tool_failed",
                        f"相似案例检索完成，获得 {len(rag_result.sources)} 条参考资料" if rag_result.ok else "相似案例检索失败")
        results = [vision_result, rag_result]
        if not rag_result.ok:
            return SkillResult(skill_name=self.name, ok=False,
                               answer="图片特征已提取，但知识库未检索到可用的相似案例。",
                               tool_results=results, evidence=vision_result.sources)

        answer = (
            "## 历史案例检索结果\n\n"
            f"{rag_result.data.get('summary', '')}\n\n"
            "> 说明：以上仅为图片特征与历史案例的相似性比对，不构成对当前图片违法或合法的认定。"
            "如需违法判定，请明确提出判定请求，并补充许可、时间、地点等证据。"
        )
        return SkillResult(
            skill_name=self.name,
            ok=True,
            answer=answer,
            tool_results=results,
            evidence=[source for result in results for source in result.sources],
            metadata={"skill_package": self.package.name, "prompt_loaded": True},
            artifacts={
                "image_case_search.visual_description": visual_description,
                "image_case_search.retrieval_query": retrieval_query,
                "image_case_search.rag_result": rag_result,
                "image_case_search.context": rag_result.data.get("context", ""),
            },
        )
